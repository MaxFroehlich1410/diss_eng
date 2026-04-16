"""Generic scaling strategies for the hybrid Krotov optimizer.

Each strategy implements :meth:`compute_scaling_factors`, which returns a
per-parameter multiplicative factor applied to gate-supported Krotov updates.
Strategies are model-agnostic; model-specific structure is exposed only through
lightweight ``parameter_metadata()`` dictionaries.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# Ordered fallback keys for locating a layer-like metadata field.
LAYER_METADATA_KEYS = ("layer", "layer_index", "sublayer", "macro_layer")

SCALING_MODES = frozenset({
    "none",
    "adaptive_clip",
    "adaptive_smooth",
    "layerwise",
    "groupwise",
    "groupwise_adaptive",
})

SCALING_APPLY_PHASES = frozenset({"online", "batch", "both"})


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def get_gate_metadata(
    model, gate_indices: np.ndarray,
) -> Optional[List[Optional[Dict]]]:
    """Extract parameter metadata entries aligned with *gate_indices*.

    Returns ``None`` if the model does not expose ``parameter_metadata()``.
    """
    if not hasattr(model, "parameter_metadata"):
        return None
    all_metadata = model.parameter_metadata()
    idx_to_meta: Dict[int, Dict] = {int(m["index"]): m for m in all_metadata}
    return [idx_to_meta.get(int(i)) for i in gate_indices]


def _resolve_metadata_field(
    metadata_entries: Optional[List[Optional[Dict]]],
    candidate_keys: Sequence[str],
) -> tuple:
    """Return ``(key, True)`` for the first key with a non-None value."""
    if metadata_entries is None:
        return None, False
    for key in candidate_keys:
        for entry in metadata_entries:
            if entry is not None and key in entry and entry[key] is not None:
                return key, True
    return None, False


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------

class KrotovScalingStrategy:
    """Base interface for Krotov update scaling strategies."""

    def compute_scaling_factors(
        self,
        raw_update: np.ndarray,
        gate_indices: np.ndarray,
        model=None,
        metadata: Optional[List[Optional[Dict]]] = None,
    ) -> np.ndarray:
        """Return multiplicative scaling factors for gate-supported parameters.

        Parameters
        ----------
        raw_update : ndarray, shape ``(n_gate,)``
            Raw gate-supported update entries.
        gate_indices : ndarray of int, shape ``(n_gate,)``
            Parameter indices corresponding to *raw_update*.
        model : optional
            The QML model instance.
        metadata : list of dict or None
            Per-gate parameter metadata aligned with *gate_indices*.

        Returns
        -------
        ndarray, shape ``(n_gate,)``
        """
        raise NotImplementedError

    def validate(self, gate_indices, model=None, metadata=None):
        """Pre-training validation hook (called once before the loop)."""


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class NoScaling(KrotovScalingStrategy):
    """Identity scaling — reproduces baseline Krotov behavior."""

    def compute_scaling_factors(self, raw_update, gate_indices, model=None, metadata=None):
        return np.ones(len(gate_indices), dtype=float)


class AdaptiveClipScaling(KrotovScalingStrategy):
    r"""Hard per-parameter clipping.

    .. math:: S_k = \min\!\bigl(1,\;\tau\,/\,(|u_k| + \varepsilon)\bigr)
    """

    def __init__(self, tau: float = 0.1, eps: float = 1e-8):
        if tau <= 0:
            raise ValueError(f"AdaptiveClipScaling requires tau > 0, got {tau}.")
        if eps < 0:
            raise ValueError(f"AdaptiveClipScaling requires eps >= 0, got {eps}.")
        self.tau = float(tau)
        self.eps = float(eps)

    def compute_scaling_factors(self, raw_update, gate_indices, model=None, metadata=None):
        magnitudes = np.abs(raw_update)
        return np.minimum(1.0, self.tau / (magnitudes + self.eps))


class AdaptiveSmoothScaling(KrotovScalingStrategy):
    r"""Smooth per-parameter damping.

    .. math:: S_k = 1\,/\,(1 + \beta\,|u_k|)
    """

    def __init__(self, beta: float = 1.0):
        if beta < 0:
            raise ValueError(f"AdaptiveSmoothScaling requires beta >= 0, got {beta}.")
        self.beta = float(beta)

    def compute_scaling_factors(self, raw_update, gate_indices, model=None, metadata=None):
        return 1.0 / (1.0 + self.beta * np.abs(raw_update))


class LayerwiseScaling(KrotovScalingStrategy):
    r"""Per-layer scaling via exponential decay or an explicit map.

    Exponential form: :math:`S_k = \gamma^{\ell(k)}`

    Parameters
    ----------
    gamma : float or None
        Decay base for the exponential form (must satisfy ``0 < gamma <= 1``).
    layer_scales : dict or None
        Explicit ``{layer_index: scale}`` mapping.
    layer_field : str or None
        Metadata field to read.  If ``None``, an ordered fallback search over
        ``LAYER_METADATA_KEYS`` is used.
    """

    def __init__(self, gamma=None, layer_scales=None, layer_field=None):
        if gamma is None and layer_scales is None:
            raise ValueError(
                "LayerwiseScaling requires either gamma (exponential decay) "
                "or layer_scales (explicit mapping)."
            )
        if gamma is not None and layer_scales is not None:
            raise ValueError("LayerwiseScaling accepts gamma or layer_scales, not both.")
        if gamma is not None and not (0 < gamma <= 1.0):
            raise ValueError(f"LayerwiseScaling requires 0 < gamma <= 1, got {gamma}.")
        self.gamma = gamma
        self.layer_scales = dict(layer_scales) if layer_scales is not None else None
        self.layer_field = layer_field

    def _resolve_field(self, metadata):
        if self.layer_field is not None:
            for entry in (metadata or []):
                if entry is not None and self.layer_field in entry and entry[self.layer_field] is not None:
                    return self.layer_field
            raise ValueError(
                f"LayerwiseScaling: requested metadata field '{self.layer_field}' "
                f"not found (with a non-None value) in any gate parameter metadata."
            )
        field, found = _resolve_metadata_field(metadata, LAYER_METADATA_KEYS)
        if not found:
            raise ValueError(
                f"LayerwiseScaling: no usable layer-like metadata field found. "
                f"Searched for: {LAYER_METADATA_KEYS}. "
                f"The model must expose at least one of these in parameter_metadata()."
            )
        return field

    def validate(self, gate_indices, model=None, metadata=None):
        if metadata is None:
            raise ValueError(
                "LayerwiseScaling requires parameter metadata, but none was provided."
            )
        self._resolve_field(metadata)

    def compute_scaling_factors(self, raw_update, gate_indices, model=None, metadata=None):
        if metadata is None:
            raise ValueError("LayerwiseScaling requires parameter metadata.")
        field = self._resolve_field(metadata)
        scales = np.ones(len(gate_indices), dtype=float)
        for i, entry in enumerate(metadata):
            if entry is None:
                raise ValueError(
                    f"LayerwiseScaling: gate parameter at position {i} "
                    f"(index {gate_indices[i]}) has no metadata."
                )
            layer_val = entry.get(field)
            if layer_val is None:
                raise ValueError(
                    f"LayerwiseScaling: gate parameter at index {gate_indices[i]} "
                    f"has no value for metadata field '{field}'."
                )
            layer_idx = int(layer_val)
            if self.gamma is not None:
                scales[i] = self.gamma ** layer_idx
            else:
                if layer_idx not in self.layer_scales:
                    raise ValueError(
                        f"LayerwiseScaling: no scale defined for layer index "
                        f"{layer_idx}. Available: {sorted(self.layer_scales.keys())}."
                    )
                scales[i] = float(self.layer_scales[layer_idx])
        return scales


class GroupwiseScaling(KrotovScalingStrategy):
    """Static per-group scaling based on a metadata label.

    Parameters
    ----------
    group_field : str
        Metadata field to read group labels from.
    group_scales : dict
        ``{label: scale_factor}`` mapping.
    default_group_scale : float or None
        Fallback scale for labels absent from *group_scales*.  If ``None``,
        missing labels raise an error.
    """

    def __init__(self, group_field: str, group_scales: dict, default_group_scale=None):
        if not group_field:
            raise ValueError("GroupwiseScaling requires a non-empty group_field.")
        if not group_scales:
            raise ValueError("GroupwiseScaling requires a non-empty group_scales mapping.")
        self.group_field = str(group_field)
        self.group_scales = dict(group_scales)
        self.default_group_scale = (
            float(default_group_scale) if default_group_scale is not None else None
        )

    def validate(self, gate_indices, model=None, metadata=None):
        if metadata is None:
            raise ValueError(
                f"GroupwiseScaling requires parameter metadata with field "
                f"'{self.group_field}', but none was provided."
            )
        has_field = any(
            entry is not None and self.group_field in entry
            for entry in metadata
        )
        if not has_field:
            raise ValueError(
                f"GroupwiseScaling: metadata field '{self.group_field}' not found "
                f"in any gate parameter metadata."
            )

    def compute_scaling_factors(self, raw_update, gate_indices, model=None, metadata=None):
        if metadata is None:
            raise ValueError("GroupwiseScaling requires parameter metadata.")
        scales = np.ones(len(gate_indices), dtype=float)
        for i, entry in enumerate(metadata):
            if entry is None:
                raise ValueError(
                    f"GroupwiseScaling: gate parameter at position {i} "
                    f"(index {gate_indices[i]}) has no metadata."
                )
            if self.group_field not in entry:
                raise ValueError(
                    f"GroupwiseScaling: gate parameter at index {gate_indices[i]} "
                    f"has no metadata field '{self.group_field}'."
                )
            label = entry[self.group_field]
            if label in self.group_scales:
                scales[i] = float(self.group_scales[label])
            elif self.default_group_scale is not None:
                scales[i] = self.default_group_scale
            else:
                raise ValueError(
                    f"GroupwiseScaling: gate parameter at index {gate_indices[i]} "
                    f"has group label '{label}' which is not in group_scales "
                    f"{set(self.group_scales.keys())} and no default_group_scale is set."
                )
        return scales


class GroupwiseAdaptiveScaling(KrotovScalingStrategy):
    r"""Product of a static groupwise factor and an adaptive factor.

    .. math:: S_k^{\text{total}} = S_k^{\text{group}} \cdot S_k^{\text{adaptive}}
    """

    def __init__(
        self,
        groupwise: GroupwiseScaling,
        adaptive: KrotovScalingStrategy,
    ):
        if not isinstance(groupwise, GroupwiseScaling):
            raise TypeError("groupwise must be a GroupwiseScaling instance.")
        if not isinstance(adaptive, (AdaptiveClipScaling, AdaptiveSmoothScaling)):
            raise TypeError(
                "adaptive must be an AdaptiveClipScaling or AdaptiveSmoothScaling instance."
            )
        self.groupwise = groupwise
        self.adaptive = adaptive

    def validate(self, gate_indices, model=None, metadata=None):
        self.groupwise.validate(gate_indices, model, metadata)
        self.adaptive.validate(gate_indices, model, metadata)

    def compute_scaling_factors(self, raw_update, gate_indices, model=None, metadata=None):
        group_factors = self.groupwise.compute_scaling_factors(
            raw_update, gate_indices, model, metadata
        )
        adaptive_factors = self.adaptive.compute_scaling_factors(
            raw_update, gate_indices, model, metadata
        )
        return group_factors * adaptive_factors


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_scaling_strategy(
    scaling_mode: str,
    scaling_config: Optional[dict] = None,
) -> KrotovScalingStrategy:
    """Construct a scaling strategy from a mode string and config dict.

    Parameters
    ----------
    scaling_mode : str
        One of ``SCALING_MODES``.
    scaling_config : dict or None
        Strategy-specific parameters.

    Returns
    -------
    KrotovScalingStrategy
    """
    if scaling_mode not in SCALING_MODES:
        raise ValueError(
            f"Unknown scaling mode '{scaling_mode}'. "
            f"Valid modes: {sorted(SCALING_MODES)}."
        )

    cfg = scaling_config or {}

    if scaling_mode == "none":
        return NoScaling()

    if scaling_mode == "adaptive_clip":
        return AdaptiveClipScaling(
            tau=cfg.get("tau", 0.1),
            eps=cfg.get("eps", 1e-8),
        )

    if scaling_mode == "adaptive_smooth":
        return AdaptiveSmoothScaling(beta=cfg.get("beta", 1.0))

    if scaling_mode == "layerwise":
        return LayerwiseScaling(
            gamma=cfg.get("gamma"),
            layer_scales=cfg.get("layer_scales"),
            layer_field=cfg.get("layer_field"),
        )

    if scaling_mode == "groupwise":
        return GroupwiseScaling(
            group_field=cfg.get("group_field", "group"),
            group_scales=cfg.get("group_scales", {}),
            default_group_scale=cfg.get("default_group_scale"),
        )

    if scaling_mode == "groupwise_adaptive":
        adaptive_mode = cfg.get("adaptive_mode", "adaptive_clip")
        adaptive_cfg = cfg.get("adaptive_config", {})
        if adaptive_mode == "adaptive_clip":
            adaptive = AdaptiveClipScaling(
                tau=adaptive_cfg.get("tau", 0.1),
                eps=adaptive_cfg.get("eps", 1e-8),
            )
        elif adaptive_mode == "adaptive_smooth":
            adaptive = AdaptiveSmoothScaling(beta=adaptive_cfg.get("beta", 1.0))
        else:
            raise ValueError(
                f"groupwise_adaptive: unknown adaptive_mode '{adaptive_mode}'. "
                f"Use 'adaptive_clip' or 'adaptive_smooth'."
            )
        groupwise = GroupwiseScaling(
            group_field=cfg.get("group_field", "group"),
            group_scales=cfg.get("group_scales", {}),
            default_group_scale=cfg.get("default_group_scale"),
        )
        return GroupwiseAdaptiveScaling(groupwise, adaptive)

    raise ValueError(f"Unhandled scaling mode: {scaling_mode}")

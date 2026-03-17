qal_MFB85BKy_FABcTQTYLnGarGpcKmZB5o81

export Q_ALCHEMY_API_KEY="qal_MFB85BKy_FABcTQTYLnGarGpcKmZB5o81"
python3 experiments/hydrogen_qalchemy/run_hydrogen_qalchemy.py

python3 experiments/hydrogen_qalchemy/run_hydrogen_qalchemy.py
  --search
  --num-qubits 10
  --n-max 5
  --min-final-fidelity 0.95
  --min-monotonic-ratio 0.98

python3 experiments/hydrogen_qalchemy/run_hydrogen_qalchemy.py --n=2 --l=0 --m=0




export Q_ALCHEMY_API_KEY="qal_MFB85BKy_FABcTQTYLnGarGpcKmZB5o81"
python3 experiments/hydrogen_qalchemy/compare_tails.py
python3 experiments/hydrogen_qalchemy/compare_tails.py --num-qubits 6 --fit-iters 300 --gamma-max 3
python3 experiments/hydrogen_qalchemy/compare_tails.py --fit-loss frobenius --no-2q

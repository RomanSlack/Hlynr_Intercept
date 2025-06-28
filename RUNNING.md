# Quick Start (Phase-2)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# train headless
python scripts/train_ppo_phase2.py
# train with live 3-D viewer
python scripts/train_ppo_phase2.py --visualize
# resume from checkpoint
python scripts/train_ppo_phase2.py --resume models/phase2/ppo_050000.pt

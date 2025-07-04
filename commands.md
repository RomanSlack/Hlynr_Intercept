tensorboard --logdir=runs --port=6006

python scripts/train_ppo_phase3_6dof.py
    - Resume: --resume=models/phase3/latest.pt
    - Visualize: --visualize (enhanced 6DOF real-time rendering)
# Single-Leg Robot Stabilization (MuJoCo + RL)

This project simulates a 3-DoF single-legged robot using **MuJoCo** and trains a reinforcement learning policy using **PPO (Proximal Policy Optimization)** to maintain upright balance under disturbances.

Key features:
- Custom MuJoCo model with hip, knee, and ankle joints
- PPO agent trained with `Stable-Baselines3`
- Disturbance injection via lateral force pushes
- Real-time evaluation and visualization with MuJoCo viewer

### Tech Stack
- Python, MuJoCo, Gymnasium
- Stable-Baselines3 (PPO)
- TensorBoard (for logging)

### To Run
```bash
# Train the policy
mjpython train.py

# Visualize the robot reacting to pushes
mjpython evaluate.py

```
[Work in progress]
<img width="1512" alt="Screenshot 2025-04-23 at 2 23 41 PM" src="https://github.com/user-attachments/assets/0aca02d7-df66-405c-abeb-4804eb1b1fd1" />

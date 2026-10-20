# Lunar Lander: PPO Reward Shaping

This is a reproducible reinforcement-learning project based on Gymnasium's `LunarLander-v3`. It compares:

- `baseline`: the simulator's original reward;
- `hacked`: an intentionally flawed per-timestep bonus that can teach the agent to hover;
- `fixed`: progress-based shaping plus a small time cost.

The project evaluates every policy using the unmodified Lunar Lander reward. A high shaped training score alone is never treated as success.

## Prerequisites

- Windows, macOS, or Linux
- Python 3.10 or newer
- A terminal with Python on `PATH`

## Install

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If Box2D fails with a `swig` error on Windows, install SWIG and repeat the last command.

## Validate setup

```powershell
python -c "from src.envs import make_lander; env=make_lander(); print(env.reset(seed=42)[0]); env.close()"
pytest
```

## Train

Start with a quick smoke run. A one-seed comparison takes **3 training runs** (one per condition). For the full experiment, train each of the 3 conditions with all 3 seeds (`42`, `314`, and `2026`): **9 training runs total**.

```powershell
# Smoke run
python -m src.train --condition baseline --timesteps 10000 --seed 42

# Full experiment: 3 conditions x 3 seeds = 9 training runs
python -m src.train --condition baseline --timesteps 1000000 --seed 42
python -m src.train --condition baseline --timesteps 1000000 --seed 314
python -m src.train --condition baseline --timesteps 1000000 --seed 2026
python -m src.train --condition hacked --timesteps 1000000 --seed 42
python -m src.train --condition hacked --timesteps 1000000 --seed 314
python -m src.train --condition hacked --timesteps 1000000 --seed 2026
python -m src.train --condition fixed --timesteps 1000000 --seed 42
python -m src.train --condition fixed --timesteps 1000000 --seed 314
python -m src.train --condition fixed --timesteps 1000000 --seed 2026
```

Each training run creates a saved model, checkpoints, TensorBoard logs, episode monitor logs, evaluation logs, and a JSON file with its configuration.

To inspect learning while training:

```powershell
tensorboard --logdir results
```

## Evaluate native task performance

These commands deliberately use an **unwrapped** environment. This avoids mistaking reward shaping for successful landing.

```powershell
python -m src.evaluate --model models/baseline_seed42/best/best_model --label baseline_seed42
python -m src.evaluate --model models/hacked_seed42/best/best_model --label hacked_seed42
python -m src.evaluate --model models/fixed_seed42/best/best_model --label fixed_seed42
```

Generate a comparison image:

```powershell
python -m src.compare `
  results/native_evaluations/baseline_seed42_summary.json `
  results/native_evaluations/hacked_seed42_summary.json `
  results/native_evaluations/fixed_seed42_summary.json
```

## Record comparable videos

Use the same seeds across conditions so the clips are directly comparable.

```powershell
python -m src.record_video --model models/baseline_seed42/best/best_model --label baseline_seed42
python -m src.record_video --model models/hacked_seed42/best/best_model --label hacked_seed42
python -m src.record_video --model models/fixed_seed42/best/best_model --label fixed_seed42
```

Videos are stored under `results/videos/`.

## Record cinematic Moon-mission videos

The cinematic renderer is a post-training visual layer: it creates a dedicated
Moon mission scene driven by the real simulation state, with a custom lander,
animated engine flames, cratered terrain, a landing pad, star field, Earth,
mission frame, and live telemetry. It changes **none** of the model inputs,
rewards, or physics.

```powershell
python -m src.record_cinematic --model models/baseline_seed42/best/best_model --label baseline_seed42 --episodes 150
python -m src.record_cinematic --model models/hacked_seed42/best/best_model --label hacked_seed42 --episodes 150
python -m src.record_cinematic --model models/fixed_seed42/best/best_model --label fixed_seed42 --episodes 150
```

Open the generated `.mp4` files in `results/cinematic_videos/`. Use matching
seeds (the defaults do) whenever you compare conditions. The cinematic recorder
defaults to 3 episodes; pass `--episodes 150` to record 150 landing attempts
for a model. For the 9-run experiment, repeat evaluation and recording for each
condition and training seed.

## How to report results

For every condition and seed, report native mean return, standard deviation, success rate (episodes with native return >= 200), and mean episode length.

| Condition | Expected observation |
| --- | --- |
| Baseline | Learns a useful landing policy. |
| Hacked | May receive high shaped reward while hovering or delaying. |
| Fixed | Regains native landing performance by rewarding progress instead of waiting. |

If the hacked condition does not visibly show the intended exploit after the full budget, adjust the `bonus` in `HoverRewardWrapper` conservatively, retrain, and document the change. Never change the unwrapped evaluation environment.

# Race Agent Lab

Train a reinforcement-learning racing agent in your browser, no installs,
then race it against your classmates in the
[City Grand Prix arena](https://city-grand-prix.vercel.app).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreedath/race-agent-lab/blob/main/train_agent.ipynb)

## How it works

1. Click the **Open in Colab** badge above.
2. Run the setup cell (about 1 minute).
3. **Design your reward function**: the notebook explains dense vs sparse
   rewards, reward scale, and the loopholes agents love to exploit. This is
   the whole assignment; everything else is plumbing.
4. Train with PPO (10 to 30 minutes on the free Colab CPU). You can stop
   early at any time and keep what you have.
5. Export: you get a small `<YourAgentName>.json` file with your policy
   weights. Download it and send it to your instructor, or drag it into the
   race arena lobby yourself.

## What's in here

```
train_agent.ipynb      the one-click Colab notebook (start here)
racing/sim/            car physics, walls, LiDAR (identical to the arena)
racing/track/          track geometry pipeline
racing/env/race_env.py Gymnasium env; takes YOUR reward function
racing/train/          PPO training + policy export
shared/                track and physics constants shared with the arena
```

The physics here is bit-for-bit the same simulation the browser arena runs,
which is why a policy trained in Colab drives identically in the race.

## Rules of the race

- Training cap: 300k steps (the notebook enforces it). Reward design beats
  brute force.
- One agent file per student, named via `AGENT_NAME` in the notebook.
- The instructor loads everyone's file into the arena lobby and runs the
  grand prix on an F1-style grid: 3 laps, first across the line wins.
  A car that gets stuck is classified DNF 45 seconds after the winner
  finishes.

## Local use (optional)

Works outside Colab too:

```bash
pip install gymnasium stable-baselines3
python -m racing.train.train_ppo --steps 100000       # sample reward
python -m racing.train.export_policy --model runs/ppo_race --name "My Racer"
```

# Snake With Brain

An advanced Deep Q-Learning Snake agent implemented with PyGame, NumPy, and
PyTorch. The project includes a dueling Q-network, Double-DQN targets,
prioritized replay, checkpoint resume, deterministic evaluation, and
headless training.

## Setup

Python 3.13 is the recommended Windows runtime for the current dependency wheels:

```powershell
py -3.13 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

The dependency file selects wheels for Python 3.13 and retains the older
compatible pins for Python versions below 3.13.

## Run training

Launch the interactive trainer:

```powershell
.\\.venv\\Scripts\\python.exe agent.py
```

For reproducible training without a window, frame-rate waits, or live plots:

```powershell
.\\.venv\\Scripts\\python.exe agent.py --headless --max-games 1000 --seed 42
```

By default, training resumes `model/checkpoint.pth`. To discard the saved
training progress and begin at game 0 with newly initialized weights, use
`--fresh`:

```powershell
.\\.venv\\Scripts\\python.exe agent.py --fresh --headless --max-games 100 --seed 42
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--headless` | Disable PyGame rendering and frame-rate waits. |
| `--fresh` | Ignore the saved checkpoint and start from game 0. |
| `--max-games N` | Stop after `N` completed games. |
| `--seed N` | Seed Python, NumPy, PyTorch, and game randomness. |
| `--eval-games N` | Run `N` greedy evaluation games during training. |
| `--eval-every N` | Evaluate after every `N` completed games. `0` evaluates every episode when evaluation is enabled. |
| `--log-file PATH` | Write structured JSONL records. Default: `runs/training.jsonl`. Pass an empty value to disable logging. |

Example experiment command:

```powershell
.\\.venv\\Scripts\\python.exe agent.py --headless --max-games 1000 --seed 42 --eval-games 10 --eval-every 50 --log-file runs/seed-42.jsonl
```

The existing `model/checkpoint.pth` is loaded automatically when present.
Training saves the current model, target model, optimizer, replay buffer,
progress counters, and checkpoint metadata after each completed game.

## Architecture and learning schedule

- State: 11 binary features covering danger, direction, and food location.
- Policy and target networks: dueling MLPs with hidden sizes `512 -> 512 -> 256` and 3 action outputs (straight, right, left).
- Targets: Double-DQN action selection with a softly updated target network.
- Replay: prioritized replay with alpha `0.6`, importance-sampling beta annealing, and a capacity of `100,000` transitions.
- Optimization: Adam, learning rate `0.0005`, gradient clipping, and batches of `512`.
- Schedule: one prioritized replay update every 4 environment frames after the initial replay batch is available. There is no additional single-transition update or episode-end duplicate update.
- Exploration: exponential decay from `1.0` toward a persistent floor of `0.05`, so exploration does not stop after 80 games.

The simulator permits moving into the departing tail cell, places food from a
bounded list of free cells, and ends an episode cleanly when the board is full.

## Checkpoints and evaluation

New checkpoints include model architecture and optimizer metadata. Resume only
loads a model when the complete model and target state dictionaries match; an
incompatible optimizer is skipped instead of producing an inconsistent
resume. Older checkpoints without metadata are still accepted when their
complete tensor and optimizer structures match.

Evaluation uses separate seeded environments and greedy actions. It restores
the training RNG states, exploration value, and model training mode after the
run, so evaluation does not perturb training.

## Experiment logs

Training writes one JSON object per line. Records include `run_started`,
`episode`, `evaluation`, and `run_finished` events with timestamps, seed,
episode score, running mean, record, epsilon, frame count, replay size, and
evaluation scores. Generated `runs/` files are ignored by Git.

## Tests

Run the regression suite in headless mode:

```powershell
$env:SDL_VIDEODRIVER = 'dummy'
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
```

The tests cover tail collision semantics, full-board food placement,
headless simulation, checkpoint compatibility, exploration scheduling,
replay scheduling, deterministic seeds, independent evaluation, and JSONL
logging.


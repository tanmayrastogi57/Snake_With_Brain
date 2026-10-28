# Snake With Brain

An advanced Deep Q-Learning Snake agent implemented with PyGame, NumPy, and
PyTorch. The project includes a dueling Q-network, Double-DQN targets,
prioritized replay, checkpoint resume, deterministic evaluation, and
headless training.

## Setup

Python 3.13 is the recommended Windows runtime for the current dependency wheels:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The dependency file selects wheels for Python 3.13 and retains the older
compatible pins for Python versions below 3.13.

## Command reference

Run these commands from the repository directory:

| Command | What it does |
| --- | --- |
| `py -3.13 -m venv .venv` | Creates the project virtual environment. |
| `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` | Installs the pinned Python/PyTorch/PyGame dependencies. |
| `.\.venv\Scripts\python.exe agent.py --help` | Shows every training option. |
| `.\.venv\Scripts\python.exe agent.py` | Starts interactive single-environment training and resumes `model/checkpoint.pth`. |
| `.\.venv\Scripts\python.exe agent.py --fresh` | Starts interactive training from new weights and game 0. |
| `.\.venv\Scripts\python.exe agent.py --headless --max-games 1000 --seed 42` | Runs one reproducible CPU training process without a Pygame window. |
| `.\.venv\Scripts\python.exe agent.py --headless --device cuda --max-games 1000` | Runs one learner on CUDA when a compatible GPU build is installed. |
| `.\.venv\Scripts\python.exe agent.py --workers 10 --headless --device auto --fresh --max-games 10000 --seed 42` | Runs 10 CPU environment workers feeding one shared learner and replay buffer. |
| `.\.venv\Scripts\python.exe agent.py --workers 10 --headless --device cuda --checkpoint runs\shared\checkpoint.pth --log-file runs\shared\training.jsonl` | Runs configurable shared-experience training with the learner on CUDA. |
| `.\.venv\Scripts\python.exe agent.py --workers 10 --headless --policy-sync-every 100` | Shares one learner while broadcasting updated policy weights every 100 learner updates. |
| `.\.venv\Scripts\python.exe agent.py --workers 4 --render-workers --max-games 1000` | Opens one visible Pygame window per worker while using one shared learner. |
| `.\.venv\Scripts\python.exe plot_training.py --log-file runs\shared\shared-training.jsonl --output runs\shared\shared-training.png` | Generates an analysis graph from a completed or currently running JSONL log. |
| `.\.venv\Scripts\python.exe plot_monitor.py --log-file runs\shared\shared-training.jsonl` | Opens a live graph window that refreshes while training writes the log. |
| `.\run_parallel_training.ps1 -Workers 10 -Games 10000 -Device auto` | Starts the shared learner with 10 configurable environment workers. |
| `.\run_parallel_training.ps1 -Workers 4 -Games 0 -Device auto -Render` | Runs indefinitely until manually stopped, with visible workers and an automatically opened graph. |
| `.\run_parallel_training.ps1 -Workers 10 -Games 10000 -Device cuda` | Starts shared training with the learner on CUDA. |
| `.\run_parallel_training.ps1 -Workers 10 -Games 10000 -Device auto -Resume` | Resumes the shared checkpoint created by the launcher. |
| `.\run_parallel_training.ps1 -Workers 4 -Games 1000 -Device auto -Render` | Shows worker windows and automatically opens the live analysis graph. |
| `.\run_parallel_training.ps1 -Workers 4 -Games 1000 -PlotEvery 10` | Refreshes the shared analysis graph every 10 completed games. |
| `.\run_parallel_training.ps1 -Workers 4 -Games 1000 -EvalGames 10 -EvalEvery 50` | Adds 10 greedy evaluation games every 50 training games to the graph and JSONL log. |
| `.\run_parallel_training.ps1 -Workers 4 -Games 1000 -Seed 2000 -RunDirectory runs\experiment-a` | Runs a shared experiment with custom worker seed base and output directory. |
| `.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"` | Checks whether the installed PyTorch build can see CUDA. |
| `$env:SDL_VIDEODRIVER = 'dummy'; .\.venv\Scripts\python.exe -m unittest discover -s tests -v` | Runs the regression test suite without opening a display. |
| `.\.venv\Scripts\python.exe -m pip check` | Checks for broken or missing Python dependencies. |
| `.\.venv\Scripts\python.exe -m compileall -q game.py model.py helper.py agent.py experiment.py parallel_training.py tests` | Compiles project Python files to catch syntax errors. |

`--workers N` means N environments contribute transitions to one shared
learner. It does not create N separate models. The PowerShell launcher uses
`runs\shared\shared-checkpoint.pth` and
`runs\shared\shared-training.jsonl` by default. It also refreshes
`runs\shared\shared-training.png` every 10 completed games.

`-Games 1000` stops normally after 1,000 completed games. Use `-Games 0` for
an unlimited run that continues until you press `Ctrl+C`.

## Run training

Launch the interactive trainer:

```powershell
.\.venv\Scripts\python.exe agent.py
```

For reproducible training without a window, frame-rate waits, or live plots:

```powershell
.\.venv\Scripts\python.exe agent.py --headless --max-games 1000 --seed 42
```

By default, training resumes `model/checkpoint.pth`. To discard the saved
training progress and begin at game 0 with newly initialized weights, use
`--fresh`:

```powershell
.\.venv\Scripts\python.exe agent.py --fresh --headless --max-games 100 --seed 42
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--headless` | Disable PyGame rendering and frame-rate waits. |
| `--fresh` | Ignore the saved checkpoint and start from game 0. |
| `--workers N` | Run `N` headless environments feeding one shared learner. Default: `1`. |
| `--render-workers` | Show one Pygame window per shared worker; use only when visual monitoring is needed. |
| `--device auto\|cpu\|cuda` | Select the learner device. `auto` uses CUDA when available. |
| `--policy-sync-every N` | Broadcast the latest learner policy to workers every `N` learner updates. |
| `--checkpoint PATH` | Shared checkpoint path written by the centralized learner. |
| `--max-games N` | Stop after `N` completed games. |
| `--seed N` | Seed Python, NumPy, PyTorch, and game randomness. |
| `--eval-games N` | Run `N` greedy evaluation games during training. |
| `--eval-every N` | Evaluate after every `N` completed games. `0` evaluates every episode when evaluation is enabled. |
| `--log-file PATH` | Write structured JSONL records. Default: `runs/training.jsonl`. Pass an empty value to disable logging. |

Example experiment command:

```powershell
.\.venv\Scripts\python.exe agent.py --headless --max-games 1000 --seed 42 --eval-games 10 --eval-every 50 --log-file runs/seed-42.jsonl
```

The existing `model/checkpoint.pth` is loaded automatically when present.
Training saves the current model, target model, optimizer, replay buffer,
progress counters, and checkpoint metadata after each completed game.

Multiple workers can share one learner and one replay buffer. Workers send
transitions to the learner, and the learner periodically broadcasts updated
policy weights back to them. Only the learner writes the checkpoint.

Shared workers are headless by default for speed. Add `-Render` to the
PowerShell launcher, or `--render-workers` to `agent.py`, to open one visible
Pygame window per worker. Closing a worker window stops the training run.

To start four environments contributing to one shared policy:

```powershell
.\run_parallel_training.ps1 -Workers 4 -Games 1000
```

Use CUDA when available:

```powershell
.\run_parallel_training.ps1 -Workers 10 -Games 10000 -Device cuda
```

Use `-Resume` to continue the shared checkpoint instead of starting fresh:

```powershell
.\run_parallel_training.ps1 -Workers 4 -Games 1000 -Resume
```

The worker count is configurable. The environment workers run on CPU while the
single learner runs on the selected CPU/GPU device. This shares experience and
updates one policy rather than producing separate models.

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

The shared trainer continuously writes an analysis graph with score/mean score,
record, epsilon, replay size, learner updates, and evaluation metrics. To
regenerate it after a run:

```powershell
.\.venv\Scripts\python.exe plot_training.py --log-file runs\shared\shared-training.jsonl --output runs\shared\shared-training.png
```

## Tests

Run the regression suite in headless mode:

```powershell
$env:SDL_VIDEODRIVER = 'dummy'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover tail collision semantics, full-board food placement,
headless simulation, checkpoint compatibility, exploration scheduling,
replay scheduling, deterministic seeds, independent evaluation, shared
parallel learning, device selection, and JSONL logging.


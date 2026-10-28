from pathlib import Path

import matplotlib

# Pygame owns the interactive window. Using a GUI Matplotlib backend or
# IPython.display alongside it can create competing event loops on Python 3.13.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(scores, mean_scores, output_path='runs/training.png'):
    """Save training progress without starting a second GUI event loop."""
    if not scores:
        return

    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.set_title('Training')
    axis.set_xlabel('Number of Games')
    axis.set_ylabel('Score')
    axis.plot(scores, label='score')
    axis.plot(mean_scores, label='mean score')
    axis.set_ylim(bottom=0)
    axis.legend()
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    plt.close(figure)


def plot_training_history(records, output_path='runs/training-analysis.png'):
    """Save a multi-panel analysis graph from JSONL training records."""
    episodes = [record for record in records if record.get('event') == 'episode']
    if not episodes:
        return False

    x = [record['game'] for record in episodes]
    scores = [record['score'] for record in episodes]
    mean_scores = [record.get('mean_score', 0) for record in episodes]
    epsilons = [record.get('epsilon', 0) for record in episodes]
    replay_sizes = [record.get('replay_size', 0) for record in episodes]
    learner_updates = [record.get('learner_updates', 0) for record in episodes]
    records_seen = [record.get('record', 0) for record in episodes]
    evaluations = [record for record in records if record.get('event') == 'evaluation']

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.suptitle('Snake With Brain Training Analysis')

    score_axis = axes[0, 0]
    score_axis.plot(x, scores, alpha=0.35, label='score')
    score_axis.plot(x, mean_scores, label='mean score')
    score_axis.plot(x, records_seen, label='record', linestyle='--')
    score_axis.set_title('Scores')
    score_axis.set_xlabel('Completed game')
    score_axis.set_ylabel('Score')
    score_axis.legend()

    axes[0, 1].plot(x, epsilons, color='tab:orange')
    axes[0, 1].set_title('Exploration')
    axes[0, 1].set_xlabel('Completed game')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].set_ylim(0, 1.05)

    replay_axis = axes[1, 0]
    replay_axis.plot(x, replay_sizes, color='tab:green', label='replay transitions')
    replay_axis.set_title('Shared learner progress')
    replay_axis.set_xlabel('Completed game')
    replay_axis.set_ylabel('Replay transitions')
    update_axis = replay_axis.twinx()
    update_axis.plot(x, learner_updates, color='tab:red', label='learner updates')
    update_axis.set_ylabel('Learner updates')

    evaluation_axis = axes[1, 1]
    if evaluations:
        evaluation_x = [record['game'] for record in evaluations]
        evaluation_axis.plot(evaluation_x, [record['mean_score'] for record in evaluations], label='evaluation mean')
        evaluation_axis.plot(evaluation_x, [record['max_score'] for record in evaluations], label='evaluation max')
        evaluation_axis.set_title('Greedy evaluation')
        evaluation_axis.legend()
    else:
        evaluation_axis.plot(x, scores, alpha=0)
        evaluation_axis.text(
            0.5,
            0.5,
            'No evaluation records\n(use --eval-games)',
            ha='center',
            va='center',
            transform=evaluation_axis.transAxes,
        )
        evaluation_axis.set_title('Greedy evaluation')
    evaluation_axis.set_xlabel('Completed game')
    evaluation_axis.set_ylabel('Score')

    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return True

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

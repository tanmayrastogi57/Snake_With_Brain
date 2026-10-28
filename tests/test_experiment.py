import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from experiment import JsonlLogger
from helper import plot, plot_training_history


class ExperimentLoggingTests(unittest.TestCase):
    def test_logger_writes_structured_jsonl_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'training.jsonl'
            logger = JsonlLogger(path, context={'seed': 7})
            logger.log('episode', game=3, score=12)
            logger.log('evaluation', mean_score=4.5)

            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

            self.assertEqual([record['event'] for record in records], ['episode', 'evaluation'])
            self.assertEqual(records[0]['seed'], 7)
            self.assertEqual(records[0]['score'], 12)
            self.assertIn('timestamp', records[1])

    def test_training_plot_is_saved_without_a_gui_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / 'training.png'

            plot([1, 2], [1, 1.5], output_path)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_training_analysis_plot_uses_episode_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / 'analysis.png'
            records = [
                {
                    'event': 'episode',
                    'game': 1,
                    'score': 0,
                    'record': 0,
                    'mean_score': 0.0,
                    'epsilon': 1.0,
                    'replay_size': 100,
                    'learner_updates': 0,
                },
                {
                    'event': 'episode',
                    'game': 2,
                    'score': 1,
                    'record': 1,
                    'mean_score': 0.5,
                    'epsilon': 0.99,
                    'replay_size': 200,
                    'learner_updates': 25,
                },
                {'event': 'evaluation', 'game': 2, 'mean_score': 1.0, 'max_score': 2},
            ]

            self.assertTrue(plot_training_history(records, output_path))
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


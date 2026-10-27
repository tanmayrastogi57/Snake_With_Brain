import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from experiment import JsonlLogger
from helper import plot


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


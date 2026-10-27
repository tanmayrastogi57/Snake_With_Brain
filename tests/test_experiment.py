import json
import tempfile
import unittest
from pathlib import Path

from experiment import JsonlLogger


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


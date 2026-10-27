import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch

from agent import Agent


class CheckpointRegressionTests(unittest.TestCase):
    def test_incompatible_optimizer_state_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pth")
            torch.manual_seed(1234)
            source = Agent(checkpoint_path=os.path.join(directory, "missing.pth"))
            torch.save(
                {
                    'model_state': source.model.state_dict(),
                    'target_model_state': source.target_model.state_dict(),
                    'optimizer_state': {'state': {}, 'param_groups': [{'params': []}]},
                },
                path,
            )

            resumed = Agent(checkpoint_path=path)

            self.assertEqual(resumed.model.state_dict().keys(), source.model.state_dict().keys())

    def test_incompatible_metadata_does_not_partially_load_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pth")
            source = Agent(checkpoint_path=os.path.join(directory, "missing.pth"))
            metadata = source._checkpoint_metadata()
            metadata['model']['hidden_sizes'] = [1, 1, 1]
            incompatible_state = {
                key: torch.ones_like(value)
                for key, value in source.model.state_dict().items()
            }
            torch.save(
                {
                    'metadata': metadata,
                    'model_state': incompatible_state,
                    'target_model_state': incompatible_state,
                    'optimizer_state': source.trainer.optimizer.state_dict(),
                },
                path,
            )

            resumed = Agent(checkpoint_path=path)

            first_parameter = next(resumed.model.parameters())
            self.assertFalse(torch.equal(first_parameter, torch.ones_like(first_parameter)))


if __name__ == "__main__":
    unittest.main()

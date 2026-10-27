import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch

from agent import Agent, EPSILON_END, EPSILON_START, evaluate, seed_everything


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

    def test_exploration_has_a_nonzero_floor(self):
        self.assertAlmostEqual(Agent.epsilon_for_game(0), EPSILON_START)
        self.assertGreater(Agent.epsilon_for_game(1000), EPSILON_END)
        self.assertGreater(Agent.epsilon_for_game(100000), 0.0)
        self.assertAlmostEqual(Agent.epsilon_for_game(100000), EPSILON_END, places=3)

    def test_fresh_agent_does_not_resume_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = os.path.join(directory, 'checkpoint.pth')
            source = Agent(checkpoint_path=checkpoint_path)
            source.n_games = 123
            source.save_checkpoint()

            fresh = Agent(checkpoint_path=checkpoint_path, resume=False)

            self.assertEqual(fresh.n_games, 0)

    def test_custom_checkpoint_path_is_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = os.path.join(directory, 'worker-1.pth')
            second_path = os.path.join(directory, 'worker-2.pth')
            first = Agent(checkpoint_path=first_path, resume=False)
            first.n_games = 11
            first.save_checkpoint()
            second = Agent(checkpoint_path=second_path, resume=False)
            second.n_games = 22
            second.save_checkpoint()

            first_resumed = Agent(checkpoint_path=first_path)
            second_resumed = Agent(checkpoint_path=second_path)

            self.assertEqual(first_resumed.n_games, 11)
            self.assertEqual(second_resumed.n_games, 22)

    def test_replay_training_waits_for_a_full_starting_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(checkpoint_path=os.path.join(directory, "missing.pth"))

            self.assertFalse(agent.train_from_replay())
            self.assertFalse(hasattr(agent, "train_short_memory"))

    def test_seeded_agent_initialization_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_everything(42)
            first = Agent(checkpoint_path=os.path.join(directory, "missing-1.pth"))
            seed_everything(42)
            second = Agent(checkpoint_path=os.path.join(directory, "missing-2.pth"))

            for first_parameter, second_parameter in zip(first.model.parameters(), second.model.parameters()):
                self.assertTrue(torch.equal(first_parameter, second_parameter))

    def test_evaluation_is_independent_of_training_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(checkpoint_path=os.path.join(directory, "missing.pth"))
            agent.n_games = 17
            before_epsilon = agent.epsilon

            result = evaluate(agent, games=1, seed=9, render=False)

            self.assertEqual(result['games'], 1)
            self.assertEqual(agent.n_games, 17)
            self.assertEqual(agent.epsilon, before_epsilon)


if __name__ == "__main__":
    unittest.main()

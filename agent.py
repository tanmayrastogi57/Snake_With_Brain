import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from game import SnakeGameAI, Direction, Point
from helper import plot
from model import DuelingQNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 512
LR = 0.0005
PRIORITIZED_ALPHA = 0.6
PRIORITIZED_BETA_START = 0.4
BETA_FRAMES = 100_000
PRIORITY_EPS = 1e-5


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Transition] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        transition = Transition(
            state=np.array(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=bool(done),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def _get_probabilities(self) -> np.ndarray:
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: len(self.buffer)]
        scaled_priorities = priorities ** self.alpha
        total = scaled_priorities.sum()
        if total == 0:
            return np.ones_like(scaled_priorities) / len(scaled_priorities)
        return scaled_priorities / total

    def _stratified_indices(self, probabilities: np.ndarray, batch_size: int) -> np.ndarray:
        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.0
        positions = (np.arange(batch_size) + np.random.random(batch_size)) / batch_size
        return np.searchsorted(cumulative, positions)

    def sample(self, batch_size: int, beta: float):
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        actual_batch = min(batch_size, len(self.buffer))
        probabilities = self._get_probabilities()
        indices = self._stratified_indices(probabilities, actual_batch)
        samples = [self.buffer[idx] for idx in indices]

        states = np.stack([t.state for t in samples])
        actions = np.array([t.action for t in samples], dtype=np.int64)
        rewards = np.array([t.reward for t in samples], dtype=np.float32)
        next_states = np.stack([t.next_state for t in samples])
        dones = np.array([t.done for t in samples], dtype=np.float32)

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(td_errors) + PRIORITY_EPS
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = error
        self.max_priority = max(self.max_priority, td_errors.max())

    def to_serializable(self) -> Dict[str, Any]:
        return {
            'buffer': [
                {
                    'state': transition.state.tolist(),
                    'action': transition.action,
                    'reward': transition.reward,
                    'next_state': transition.next_state.tolist(),
                    'done': transition.done,
                }
                for transition in self.buffer
            ],
            'priorities': self.priorities[: len(self.buffer)].tolist(),
            'pos': self.pos,
            'max_priority': self.max_priority,
        }

    def load_serializable(self, data: Dict[str, Any]) -> None:
        buffer_data = data.get('buffer', [])
        self.buffer = [
            Transition(
                state=np.array(item['state'], dtype=np.float32),
                action=int(item['action']),
                reward=float(item['reward']),
                next_state=np.array(item['next_state'], dtype=np.float32),
                done=bool(item['done']),
            )
            for item in buffer_data
        ]

        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        priorities = data.get('priorities', [])
        priority_array = np.array(priorities, dtype=np.float32)
        length = min(len(priority_array), len(self.buffer))
        if length:
            self.priorities[:length] = priority_array[:length]
        self.max_priority = max(float(data.get('max_priority', 1.0)), PRIORITY_EPS)
        if len(self.buffer) > length:
            self.priorities[length: len(self.buffer)] = self.max_priority
        self.pos = data.get('pos', len(self.buffer) % self.capacity)


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.frame_count = 0
        self.memory = PrioritizedReplayBuffer(MAX_MEMORY, PRIORITIZED_ALPHA)
        self.model = DuelingQNet(11, [512, 512, 256], 3)
        self.target_model = DuelingQNet(11, [512, 512, 256], 3)
        self.trainer = QTrainer(
            self.model,
            self.target_model,
            lr=LR,
            gamma=self.gamma,
            target_update_tau=0.01,
            target_update_interval=1,
        )
        self.record = 0
        self.total_score = 0
        self.plot_scores = []
        self.plot_mean_scores = []
        self.checkpoint_path = os.path.join('model', 'checkpoint.pth')
        self._load_checkpoint()

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return

        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))

        def _load_state(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor], name: str) -> bool:
            if not state_dict:
                return False
            current_state = module.state_dict()
            compatible_state = {}
            for key, value in state_dict.items():
                if key in current_state and current_state[key].shape == value.shape:
                    compatible_state[key] = value
            if not compatible_state:
                print(f"Skipping incompatible {name} checkpoint (architecture mismatch)")
                return False
            missing = set(current_state.keys()) - set(compatible_state.keys())
            unexpected = set(state_dict.keys()) - set(compatible_state.keys())
            if missing or unexpected:
                print(
                    f"Partially loaded {name} checkpoint: "
                    f"skipped {len(missing)} missing and {len(unexpected)} unexpected parameters"
                )
            module.load_state_dict({**current_state, **compatible_state})
            return True

        model_state = checkpoint.get('model_state')
        model_loaded = _load_state(self.model, model_state or {}, 'model')
        if model_loaded:
            self.target_model.load_state_dict(self.model.state_dict())

        target_state = checkpoint.get('target_model_state')
        _load_state(self.target_model, target_state or {}, 'target model')
        optimizer_state = checkpoint.get('optimizer_state')
        if optimizer_state is not None:
            self.trainer.optimizer.load_state_dict(optimizer_state)

        memory = checkpoint.get('memory')
        if memory:
            if isinstance(memory, dict):
                self.memory.load_serializable(memory)
            else:
                for item in memory:
                    state, action, reward, next_state, done = item
                    action_idx = int(np.argmax(action)) if isinstance(action, (list, tuple, np.ndarray)) else int(action)
                    self.memory.add(state, action_idx, reward, next_state, done)

        self.n_games = checkpoint.get('n_games', self.n_games)
        self.record = checkpoint.get('record', self.record)
        self.total_score = checkpoint.get('total_score', self.total_score)
        self.plot_scores = checkpoint.get('plot_scores', self.plot_scores)
        self.plot_mean_scores = checkpoint.get('plot_mean_scores', self.plot_mean_scores)
        self.frame_count = checkpoint.get('frame_count', self.frame_count)

        print(f"Loaded checkpoint with {self.n_games} games played and record {self.record}")

    def save_checkpoint(self):
        checkpoint = {
            'model_state': self.model.state_dict(),
            'target_model_state': self.target_model.state_dict(),
            'optimizer_state': self.trainer.optimizer.state_dict(),
            'memory': self.memory.to_serializable(),
            'n_games': self.n_games,
            'record': self.record,
            'total_score': self.total_score,
            'plot_scores': self.plot_scores,
            'plot_mean_scores': self.plot_mean_scores,
            'frame_count': self.frame_count,
        }

        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        torch.save(checkpoint, self.checkpoint_path)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        action_idx = int(np.argmax(action))
        self.memory.add(state, action_idx, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) == 0:
            return

        beta = min(1.0, PRIORITIZED_BETA_START + self.frame_count * (1.0 - PRIORITIZED_BETA_START) / BETA_FRAMES)
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(BATCH_SIZE, beta)
        td_errors = self.trainer.train_step(states, actions, rewards, next_states, dones, weights=weights)
        self.memory.update_priorities(indices, td_errors)

    def train_short_memory(self, state, action, reward, next_state, done):
        action_idx = int(np.argmax(action))
        self.trainer.train_step(
            np.expand_dims(np.asarray(state, dtype=np.float32), axis=0),
            np.array([action_idx], dtype=np.int64),
            np.array([reward], dtype=np.float32),
            np.expand_dims(np.asarray(next_state, dtype=np.float32), axis=0),
            np.array([done], dtype=np.float32),
        )

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            prediction = self.model(state0)[0]
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    agent = Agent()
    game = SnakeGameAI()
    plot_scores = agent.plot_scores
    plot_mean_scores = agent.plot_mean_scores
    total_score = agent.total_score
    record = agent.record
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.frame_count += 1

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            agent.record = record

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            agent.plot_scores = plot_scores
            agent.plot_mean_scores = plot_mean_scores
            agent.total_score = total_score
            agent.save_checkpoint()


if __name__ == '__main__':
    train()

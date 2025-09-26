import os
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.optim as optim


class Linear_QNet(nn.Module):
    """Legacy two-layer MLP network retained for backwards compatibility."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name: str = 'model.pth') -> None:
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class DuelingQNet(nn.Module):
    """Deeper dueling network that learns separate value and advantage streams."""

    def __init__(self, input_size: int, hidden_sizes: Iterable[int], output_size: int):
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")

        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        self.feature_layer = nn.Sequential(*layers)
        last_hidden = hidden_sizes[-1]
        self.value_stream = nn.Sequential(
            nn.Linear(last_hidden, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(last_hidden, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values

    def save(self, file_name: str = 'model.pth') -> None:
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(
        self,
        model: nn.Module,
        target_model: nn.Module,
        lr: float,
        gamma: float,
        target_update_tau: float = 0.005,
        target_update_interval: int = 1,
        max_grad_norm: Optional[float] = 10.0,
    ):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.target_update_tau = target_update_tau
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self._updates = 0

        # Ensure the target network starts as an exact copy of the policy network.
        self.target_model.load_state_dict(self.model.state_dict())
        for param in self.target_model.parameters():
            param.requires_grad = False

    def _soft_update_target(self) -> None:
        tau = self.target_update_tau
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train_step(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        weights=None,
    ):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        if weights is None:
            weights_tensor = torch.ones_like(reward)
        else:
            weights_tensor = torch.tensor(weights, dtype=torch.float32)

        q_values = self.model(state)
        current_q = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_actions = self.model(next_state).argmax(dim=1, keepdim=True)
            next_q = self.target_model(next_state).gather(1, next_actions).squeeze(-1)
            target_q = reward + self.gamma * next_q * (1 - done)

        td_errors = target_q - current_q
        loss = (weights_tensor * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        self._updates += 1
        if self._updates % self.target_update_interval == 0:
            self._soft_update_target()

        return td_errors.detach().cpu().numpy()




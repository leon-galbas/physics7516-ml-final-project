import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | list[int],
        output_dim: int,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim for _ in range(num_hidden_layers)]
        elif isinstance(hidden_dim, list):
            if not len(hidden_dim) == num_hidden_layers:
                raise ValueError(
                    f"The length of 'hidden_dim' ({len(hidden_dim)}) must match 'num_hidden_layers' ({num_hidden_layers}). "
                )
        else:
            raise ValueError(
                f"The 'hidden_dim' must be of type 'int' or 'list[int]'. Found '{type(hidden_dim)}'."
            )

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim[-1], output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

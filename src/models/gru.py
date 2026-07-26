from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout=0.1,
        predict_endpoint: bool = False,
        endpoint_dim: int = 2,
    ):
        super().__init__()

        self.predict_endpoint = predict_endpoint

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        if predict_endpoint:
            self.endpoint_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, endpoint_dim),
            )

    def forward(self, x, lengths=None):
        # GRU expects (batch_size, n_timepoints, n_features)
        x = x.permute(0, 2, 1)

        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
        else:
            packed = x

        _, h_n = self.gru(packed)
        h = h_n[-1]

        launch_params = self.param_head(h)

        if self.predict_endpoint:
            endpoint = self.endpoint_head(h)
            return launch_params, endpoint

        return launch_params

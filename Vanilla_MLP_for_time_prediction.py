import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VanillaTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout=0.0,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.encoder(x)
        return x.mean(dim=1)


class TimePredictionMLP(nn.Module):
    def __init__(self, d_model, hidden_dims):
        super().__init__()

        layers = []
        current_dim = d_model

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SchedulingTimePredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        mlp_hidden_dims,
    ):
        super().__init__()

        self.transformer = VanillaTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )

        self.mlp = TimePredictionMLP(
            d_model=d_model,
            hidden_dims=mlp_hidden_dims,
        )

    def forward(self, schedule):
        representation = self.transformer(schedule)
        return self.mlp(representation)


def load_weights(
    model,
    vanilla_weights_path,
    mlp_weights_path,
):
    vanilla_checkpoint = torch.load(
        vanilla_weights_path,
        map_location=DEVICE,
        weights_only=False,
    )

    mlp_checkpoint = torch.load(
        mlp_weights_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(vanilla_checkpoint, dict) and "state_dict" in vanilla_checkpoint:
        vanilla_checkpoint = vanilla_checkpoint["state_dict"]

    if isinstance(mlp_checkpoint, dict) and "state_dict" in mlp_checkpoint:
        mlp_checkpoint = mlp_checkpoint["state_dict"]

    model.transformer.load_state_dict(
        vanilla_checkpoint,
        strict=True,
    )

    model.mlp.load_state_dict(
        mlp_checkpoint,
        strict=True,
    )

    return model


def predict_project_time(
    model,
    candidate_schedule,
):
    model.eval()

    schedule = torch.as_tensor(
        candidate_schedule,
        dtype=torch.float32,
        device=DEVICE,
    )

    if schedule.ndim == 2:
        schedule = schedule.unsqueeze(0)

    with torch.no_grad():
        predicted_time = model(schedule)

    return predicted_time.squeeze().item()


def main():
    # M: number of skills
    # P: number of employees
    # d_model: Vanilla Transformer embedding dimension
    # nhead: number of Transformer attention heads
    # num_layers: number of Transformer encoder layers
    # dim_feedforward: Transformer feed-forward dimension
    # mlp_hidden_dims: hidden-layer dimensions of the trained MLP

    M = None
    P = None

    d_model = None
    nhead = None
    num_layers = None
    dim_feedforward = None
    mlp_hidden_dims = None

    input_dim = 1 + M + P

    vanilla_weights_path = "vanilla_transformer_weights.pt"
    mlp_weights_path = "time_prediction_mlp_weights.pt"

    model = SchedulingTimePredictor(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        mlp_hidden_dims=mlp_hidden_dims,
    ).to(DEVICE)

    model = load_weights(
        model,
        vanilla_weights_path,
        mlp_weights_path,
    )

    # Each task-day vector:
    # [h(T_i,k), C_k(T_i,1), ..., C_k(T_i,M),
    #  epsilon_k(T_i,1), ..., epsilon_k(T_i,P)]

    candidate_schedule = [
        # Scheduling-instance vectors
    ]

    predicted_time = predict_project_time(
        model,
        candidate_schedule,
    )

    print(f"Predicted project time: {predicted_time:.4f}")


if __name__ == "__main__":
    main()

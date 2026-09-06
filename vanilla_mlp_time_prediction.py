import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VANILLA_WEIGHTS_PATH = "vanilla_transformer_weights.pt"
MLP_WEIGHTS_PATH = "time_prediction_mlp_weights.pt"

M = 2
P = 3

INPUT_DIM = 1 + M + P
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512

MLP_HIDDEN_DIMS = (256, 128)


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
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
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
            input_dim=d_model,
            hidden_dims=mlp_hidden_dims,
        )

    def forward(self, schedule):
        representation = self.transformer(schedule)
        predicted_time = self.mlp(representation)
        return predicted_time


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint

    raise TypeError("Unsupported checkpoint format.")


def load_weights(model):
    vanilla_checkpoint = torch.load(
        VANILLA_WEIGHTS_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    mlp_checkpoint = torch.load(
        MLP_WEIGHTS_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    vanilla_state = extract_state_dict(vanilla_checkpoint)
    mlp_state = extract_state_dict(mlp_checkpoint)

    transformer_state = {}

    for key, value in vanilla_state.items():
        if key.startswith("transformer."):
            key = key[len("transformer."):]
        transformer_state[key] = value

    mlp_network_state = {}

    for key, value in mlp_state.items():
        if key.startswith("mlp."):
            key = key[len("mlp."):]
        if key.startswith("network."):
            mlp_network_state[key] = value
        else:
            mlp_network_state[key] = value

    model.transformer.load_state_dict(
        transformer_state,
        strict=True,
    )

    model.mlp.load_state_dict(
        mlp_network_state,
        strict=True,
    )

    return model


def predict_time(model, schedule):
    model.eval()

    schedule = torch.as_tensor(
        schedule,
        dtype=torch.float32,
        device=DEVICE,
    )

    if schedule.ndim == 2:
        schedule = schedule.unsqueeze(0)

    with torch.no_grad():
        prediction = model(schedule)

    return prediction.squeeze().item()


def main():
    model = SchedulingTimePredictor(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        mlp_hidden_dims=MLP_HIDDEN_DIMS,
    ).to(DEVICE)

    model = load_weights(model)

    candidate_schedule = [
        [1, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1],
    ]

    predicted_time = predict_time(
        model,
        candidate_schedule,
    )

    print(f"Predicted project time: {predicted_time:.4f}")


if __name__ == "__main__":
    main()

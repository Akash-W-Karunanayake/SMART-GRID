import json
from pathlib import Path
import numpy as np
import joblib
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]


class CheckpointTransformer(nn.Module):
    def __init__(self, n_features: int, horizon: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float, pe_len: int):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=pe_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        #  match checkpoint: head.0 = LayerNorm(32), head.1 = Linear(32,32), head.4 = Linear(32,H)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),        # head.0.weight shape [d_model]
            nn.Linear(d_model, d_model),  # head.1.weight shape [d_model, d_model]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),  # head.4
        )

    def forward(self, x):
        z = self.input_proj(x)
        z = self.pos_enc(z)
        z = self.encoder(z)
        z_last = z[:, -1, :]
        return self.head(z_last)

class NetLoadForecaster:
    def __init__(self, artifacts_dir: str = "artifacts_f10"):
        p = Path(__file__).resolve().parent.parent / artifacts_dir

        self.meta = json.loads((p / "meta.json").read_text())
        self.L = int(self.meta.get("L", 192))
        self.H = int(self.meta.get("H", 96))
        self.n_features = int(self.meta.get("n_features", 4))

        self.x_scaler = joblib.load(p / "x_scaler.pkl")
        self.y_scaler = joblib.load(p / "y_scaler.pkl")

        ckpt = torch.load(p / "best_transformer_cpu.pt", map_location="cpu")
        state = ckpt["model_state"]

        d_model = state["input_proj.weight"].shape[0]  # 32
        dim_ff = state["encoder.layers.0.linear1.weight"].shape[0]  # 64
        num_layers = max(int(k.split(".")[2]) for k in state.keys() if k.startswith("encoder.layers.")) + 1
        pe_len = state["pos_enc.pe"].shape[1]  # 197

        self.model = CheckpointTransformer(
            n_features=self.n_features,
            horizon=self.H,
            d_model=d_model,
            nhead=4,
            num_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=0.1,
            pe_len=pe_len,
        )

        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.feature_order = ["NetLoad_MW", "ALLSKY_SFC_SW_DWN", "T2M", "WS10M"]

    def predict(self, X_window: np.ndarray) -> np.ndarray:
        if X_window.shape != (self.L, self.n_features):
            raise ValueError(f"Expected {(self.L, self.n_features)} got {X_window.shape}")

        Xs = self.x_scaler.transform(X_window.reshape(-1, self.n_features)).reshape(1, self.L, self.n_features)
        X_tensor = torch.tensor(Xs, dtype=torch.float32)

        with torch.no_grad():
            y_scaled = self.model(X_tensor).cpu().numpy().reshape(-1, 1)

        y_mw = self.y_scaler.inverse_transform(y_scaled).reshape(-1)
        return y_mw
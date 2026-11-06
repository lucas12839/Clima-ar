from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class RMA10SpatialTemporalEncoder(nn.Module):
    """
    Encoder spatiotemporal RMA10 para ClimaAR 3.1.

    Entrada:
        x: [B, T, C, H, W]
           C=1 -> dBZ
           C=2 -> dBZ + máscara de validez

        elapsed_minutes:
            [B, T] tiempo relativo de cada frame respecto
            al frame más reciente.

    Salida:
        feature_map:
            representación espacial del último estado.

        temporal_context:
            representación temporal global.

        motion_features:
            representación aprendida del movimiento/evolución.

        evolution_features:
            representación de crecimiento/debilitamiento.

        multiscale_features:
            representación espacial multiescala.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        feature_channels: int = 128,
        num_heads: int = 8,
    ):
        super().__init__()

        if in_channels < 1:
            raise ValueError("in_channels debe ser >= 1")

        self.in_channels = in_channels
        self.feature_channels = feature_channels

        # Entrada por frame.
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            ResidualBlock(base_channels),
        )

        # Escala 1.
        self.scale1 = nn.Sequential(
            nn.Conv2d(
                base_channels,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualBlock(64),
        )

        # Escala 2.
        self.scale2 = nn.Sequential(
            nn.Conv2d(
                64,
                feature_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
            ResidualBlock(feature_channels),
        )

        # Proyección temporal de los tiempos reales.
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, feature_channels),
        )

        # Atención temporal.
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_channels,
            num_heads=num_heads,
            batch_first=True,
        )

        self.temporal_norm = nn.LayerNorm(feature_channels)

        # Atención espacial-temporal sobre los mapas.
        self.spatial_temporal_fusion = nn.Sequential(
            nn.Conv2d(
                feature_channels * 2,
                feature_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
            ResidualBlock(feature_channels),
        )

        # Cabeza de movimiento.
        self.motion_head = nn.Sequential(
            nn.Linear(feature_channels * 2, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # Cabeza de evolución/intensificación.
        self.evolution_head = nn.Sequential(
            nn.Linear(feature_channels * 2, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.output_norm = nn.LayerNorm(feature_channels)

    def forward(
        self,
        x: torch.Tensor,
        elapsed_minutes: torch.Tensor | None = None,
    ):
        if x.ndim != 5:
            raise ValueError(
                "Entrada esperada [B,T,C,H,W], "
                f"recibida {tuple(x.shape)}"
            )

        batch, time_steps, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Se esperaban {self.in_channels} canales, "
                f"recibidos {channels}"
            )

        # Procesamiento espacial independiente por frame.
        frames = x.reshape(
            batch * time_steps,
            channels,
            height,
            width,
        )

        frames = self.stem(frames)
        scale1 = self.scale1(frames)
        scale2 = self.scale2(scale1)

        _, feature_channels, h2, w2 = scale2.shape

        maps = scale2.reshape(
            batch,
            time_steps,
            feature_channels,
            h2,
            w2,
        )

        # Representación global de cada frame.
        tokens = maps.mean(dim=(3, 4))

        # Tiempo real entre observaciones.
        if elapsed_minutes is None:
            elapsed_minutes = torch.arange(
                time_steps - 1,
                -1,
                -1,
                device=x.device,
                dtype=x.dtype,
            ).unsqueeze(0).expand(batch, -1)

        if elapsed_minutes.shape != (batch, time_steps):
            raise ValueError(
                "elapsed_minutes debe tener forma "
                f"[B,T], recibido {tuple(elapsed_minutes.shape)}"
            )

        time_features = self.time_embedding(
            elapsed_minutes.unsqueeze(-1)
        )

        tokens = tokens + time_features

        # Atención temporal.
        attended, _ = self.temporal_attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )

        temporal_context = self.temporal_norm(
            attended[:, -1]
        )

        # Último mapa espacial.
        latest_map = maps[:, -1]

        # Contexto temporal expandido espacialmente.
        temporal_map = temporal_context[:, :, None, None].expand(
            -1,
            -1,
            h2,
            w2,
        )

        fused_map = self.spatial_temporal_fusion(
            torch.cat(
                [
                    latest_map,
                    temporal_map,
                ],
                dim=1,
            )
        )

        # Evolución temporal entre estados.
        if time_steps > 1:
            previous_context = attended[:, :-1].mean(dim=1)
        else:
            previous_context = attended[:, -1]

        evolution_delta = temporal_context - previous_context

        motion_input = torch.cat(
            [
                temporal_context,
                evolution_delta,
            ],
            dim=1,
        )

        motion_features = self.motion_head(
            motion_input
        )

        evolution_features = self.evolution_head(
            motion_input
        )

        temporal_context = self.output_norm(
            temporal_context
        )

        return {
            "feature_map": fused_map,
            "temporal_context": temporal_context,
            "motion_features": motion_features,
            "evolution_features": evolution_features,
            "multiscale_features": {
                "scale1": scale1.reshape(
                    batch,
                    time_steps,
                    64,
                    scale1.shape[-2],
                    scale1.shape[-1],
                ),
                "scale2": maps,
            },
        }


def build_model(
    in_channels: int = 2,
    base_channels: int = 32,
    feature_channels: int = 128,
):
    return RMA10SpatialTemporalEncoder(
        in_channels=in_channels,
        base_channels=base_channels,
        feature_channels=feature_channels,
    )


if __name__ == "__main__":
    model = build_model()

    x = torch.randn(
        1,
        6,
        2,
        64,
        64,
    )

    elapsed = torch.tensor(
        [[25.0, 20.0, 15.0, 10.0, 5.0, 0.0]]
    )

    with torch.no_grad():
        output = model(
            x,
            elapsed_minutes=elapsed,
        )

    print("RMA10 SPATIOTEMPORAL ENCODER 3.1 OK")
    print("input:", tuple(x.shape))
    print(
        "feature_map:",
        tuple(output["feature_map"].shape),
    )
    print(
        "temporal_context:",
        tuple(output["temporal_context"].shape),
    )
    print(
        "motion_features:",
        tuple(output["motion_features"].shape),
    )
    print(
        "evolution_features:",
        tuple(output["evolution_features"].shape),
    )
    print(
        "scale1:",
        tuple(output["multiscale_features"]["scale1"].shape),
    )
    print(
        "scale2:",
        tuple(output["multiscale_features"]["scale2"].shape),
    )

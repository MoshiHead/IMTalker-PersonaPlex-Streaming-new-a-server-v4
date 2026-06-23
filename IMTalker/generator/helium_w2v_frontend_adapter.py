from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


HELIUM_DIM = 4096
W2V_DIM = 768


class HeliumToWav2VecFrontendAdapter(nn.Module):
    """Map Moshi-side Helium hidden states to Wav2Vec2 projected frontend features.

    The output contract is the Wav2Vec2 internal feature_projection output:
    [B, T_w2v, 768], before Wav2Vec2 positional convolution and Transformer.
    """

    def __init__(
        self,
        helium_dim: int = HELIUM_DIM,
        w2v_dim: int = W2V_DIM,
        num_layers: int = 6,
        nhead: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        conv_kernel: int = 128,
        conv_groups: int = 16,
    ):
        super().__init__()
        if w2v_dim % nhead != 0:
            raise ValueError(f"w2v_dim={w2v_dim} must be divisible by nhead={nhead}")
        if w2v_dim % conv_groups != 0:
            raise ValueError(f"w2v_dim={w2v_dim} must be divisible by conv_groups={conv_groups}")

        self.helium_dim = int(helium_dim)
        self.w2v_dim = int(w2v_dim)
        self.num_layers = int(num_layers)

        self.input_norm = nn.LayerNorm(helium_dim)
        self.input_proj = nn.Linear(helium_dim, w2v_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.conv_kernel = int(conv_kernel)
        self.pos_conv = nn.Conv1d(
            w2v_dim,
            w2v_dim,
            kernel_size=self.conv_kernel,
            padding=self.conv_kernel // 2,
            groups=conv_groups,
        )
        self.pos_act = nn.GELU()
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=w2v_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(w2v_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        helium: torch.Tensor,
        target_len: int,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if helium.ndim != 3:
            raise ValueError(f"Expected helium [B, T, C], got {tuple(helium.shape)}")
        if helium.shape[-1] != self.helium_dim:
            raise ValueError(f"Expected helium dim {self.helium_dim}, got {helium.shape[-1]}")
        target_len = int(target_len)
        if target_len <= 0:
            raise ValueError(f"target_len must be positive, got {target_len}")

        # Student branch: Helium 12.5 Hz -> Wav2Vec2 frontend target length.
        x = F.interpolate(
            helium.transpose(1, 2).float(),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        x = self.input_dropout(self.input_proj(self.input_norm(x)))
        pos = self.pos_conv(x.transpose(1, 2))
        pos = pos[:, :, :target_len].transpose(1, 2)
        x = x + self.pos_dropout(self.pos_act(pos))
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.final_norm(x)


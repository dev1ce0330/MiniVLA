import torch
import torch.nn as nn

class MLPProjector(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(
                input_dim,
                output_dim,
            ),
            nn.GELU(),
            nn.Linear(
                output_dim,
                output_dim,
            ),
        )

    def forward(
            self,
            vision_features: torch.Tensor,
    ) -> torch.Tensor:

        projected_features = self.projector(
            vision_features
        )

        return projected_features
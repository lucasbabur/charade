"""Export the trained network to ONNX with a dynamic batch dimension."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from charade.features.spec import Encoded


def export_onnx(model: nn.Module, sample: Encoded, path: Path) -> None:
    """Write `model` (CPU, eval mode) to `path`; inputs `categorical` int64 [n, F], `dense` float32 [n, D]."""
    model = model.cpu().eval()
    cat = torch.from_numpy(sample.categorical[:8])
    dense = torch.from_numpy(sample.dense[:8])
    batch = torch.export.Dim("n")
    torch.onnx.export(
        model,
        (cat, dense),
        str(path),
        input_names=["categorical", "dense"],
        output_names=["logit"],
        dynamic_shapes={"categorical": {0: batch}, "dense": {0: batch}},
        opset_version=18,
        dynamo=True,
        optimize=True,
        external_data=False,
        verbose=False,
    )


@torch.no_grad()
def torch_logits(model: nn.Module, data: Encoded) -> np.ndarray:
    """CPU reference logits for the parity check."""
    model = model.cpu().eval()
    return model(torch.from_numpy(data.categorical), torch.from_numpy(data.dense)).numpy().astype(np.float64)

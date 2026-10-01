"""ONNX scorer: encoded features -> calibrated pCTR. Loads `model.onnx`, `feature_spec.json`, `calibrator.json`."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from charade.features.spec import Encoded, FeatureSpec
from charade.scoring.calibration import Calibrator

MODEL_FILE = "model.onnx"
SPEC_FILE = "feature_spec.json"
CALIBRATOR_FILE = "calibrator.json"


class Scorer:
    """Single-threaded ONNX Runtime session (one per worker process; batching happens per request)."""

    def __init__(self, directory: Path) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(directory / MODEL_FILE), options, providers=["CPUExecutionProvider"])
        self.spec = FeatureSpec.model_validate_json((directory / SPEC_FILE).read_text())
        self.calibrator = Calibrator.model_validate_json((directory / CALIBRATOR_FILE).read_text())

    def logits(self, data: Encoded) -> npt.NDArray[np.float64]:
        """Raw model logits."""
        out = self.session.run(["logit"], {"categorical": data.categorical, "dense": data.dense})
        return np.asarray(out[0], dtype=np.float64)

    def pctr(self, data: Encoded) -> npt.NDArray[np.float64]:
        """Calibrated click probabilities."""
        return self.calibrator.apply(self.logits(data))

from dataclasses import asdict, dataclass, field
import json


@dataclass
class DOMEHyperParams:
    # stage (a): critical decoder layers
    layers: list = field(default_factory=lambda: [14, 15, 16, 17, 18])
    module_tmp: str = 'decoder.block.{}.layer.2.DenseReluDense'

    # stage (b): edit vectors with hybrid labels
    v_lr: float = 0.5
    v_num_grad_steps: int = 400
    v_weight_decay: float = 0.2
    max_delta_norm: float = 8000.0
    lambda_start: float = 0.3
    lambda_end: float = 1.0
    lambda_steps: int = 50
    early_stop_loss: float = 0.05
    v_min_prob: float = 0.9  # stop only when every target token has at least this probability
    z_batch_size: int = 256

    # stage (c): layer-wise weight update
    mom2_update_weight: float = 300.0
    mom2_n_samples: int = 400000
    mom2_batch_size: int = 64
    key_batch_size: int = 64
    ridge: float = 1e-4

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def to_dict(self):
        return asdict(self)

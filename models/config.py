import json
import os
from dataclasses import asdict, dataclass, fields

try:
    import yaml
except ImportError:
    yaml = None

DEFAULT_CONFIG_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "model_config.yaml",
)


@dataclass
class DeepSeekConfig:
    # Model architecture
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8

    # MLA configuration
    kv_lora_rank: int = 128
    q_lora_rank: int = 192
    rope_dim: int = 32

    # MoE configuration
    n_experts: int = 8
    n_experts_per_token: int = 2
    expert_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 768
    use_shared_expert: bool = True

    # MTP configuration
    mtp_num_heads: int = 1

    # Training parameters
    dropout: float = 0.1
    bias: bool = True
    aux_loss_weight: float = 0.0
    mtp_loss_weight: float = 0.3

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path):
        if yaml is None:
            raise ImportError("pyyaml is required to load configs from YAML (pip install pyyaml)")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    @classmethod
    def default(cls):
        """Single source of truth for the model architecture.

        Loads configs/model_config.yaml when present so the authoritative
        config lives in one editable file; falls back to the dataclass
        defaults above otherwise (e.g. in environments without pyyaml).
        """
        if yaml is not None and os.path.exists(DEFAULT_CONFIG_YAML):
            return cls.from_yaml(DEFAULT_CONFIG_YAML)
        return cls()

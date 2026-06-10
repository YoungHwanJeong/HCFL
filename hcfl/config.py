# Experiment configuration dataclasses with YAML load/save.
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict

import yaml


@dataclass
class DataConfig:
    name: str = "Caco2_Wang"
    task: str = "regression"
    num_classes: int = 1
    data_root: str = "data"
    dirichlet_alpha: float = 0.5
    min_samples_per_client: int = 100
    max_samples_per_client: int = 500
    label_noise_std_frac: float = 0.10
    test_frac: float = 0.1
    val_frac: float = 0.1
    max_smiles_len: int = 128
    seed: int = 42


@dataclass
class ModelConfig:
    backbone_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    embedder_name: str = "DeepChem/ChemBERTa-5M-MLM"
    head_hidden_dim: int = 256
    head_dropout: float = 0.1
    freeze_backbone_layers: int = 0


@dataclass
class FederatedConfig:
    num_clients: int = 50
    participation_rate: float = 0.20
    num_rounds: int = 100
    local_epochs: int = 5
    learning_rate: float = 2e-5
    batch_size: int = 16
    deadline: float = 1.0
    cpu_cores_range: tuple = (2, 16)
    memory_gb_range: tuple = (4, 16)
    gpu_frac_range: tuple = (0.10, 0.20)


@dataclass
class ContributionConfig:
    estimator: str = "hybrid"
    alpha_init: float = 0.5
    alpha_min: float = 0.2
    alpha_max: float = 0.8
    adapt_alpha: bool = True
    fim_batch_size: int = 32
    fim_max_batches: int = 8
    chi2_quantile: float = 0.95
    density_threshold_quantile: float = 0.30
    sparsity_power: float = 2.0
    qmc_samples: int = 512
    cov_shrinkage: float = 1e-2
    shape_penalty_weight: float = 0.10
    vol_penalty_weight: float = 0.50
    eps: float = 1e-8


@dataclass
class SelectionConfig:
    delta: float = 1.0
    delta_in: float = 0.5
    ema_gamma: float = 0.9
    r_split: int = 3
    tau_split: float = 0.5
    failure_ema: float = 0.5
    max_splits: int = 8


@dataclass
class HCFLConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    contribution: ContributionConfig = field(default_factory=ContributionConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> "HCFLConfig":
        def build(klass, sub):
            fields = {f.name for f in dataclasses.fields(klass)}
            return klass(**{k: v for k, v in (sub or {}).items() if k in fields})

        top = {k: v for k, v in d.items()
               if k not in {"data", "model", "federated", "contribution", "selection"}}
        return cls(
            data=build(DataConfig, d.get("data")),
            model=build(ModelConfig, d.get("model")),
            federated=build(FederatedConfig, d.get("federated")),
            contribution=build(ContributionConfig, d.get("contribution")),
            selection=build(SelectionConfig, d.get("selection")),
            **top,
        )

    @classmethod
    def load(cls, path: str) -> "HCFLConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

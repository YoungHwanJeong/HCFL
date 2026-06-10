#!/usr/bin/env python
# Regenerate YAML configs from dataclass defaults.  Run: python configs/_generate_configs.py
import os

from hcfl.config import HCFLConfig

HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "caco2":    dict(name="Caco2_Wang", task="regression", num_classes=1),
    "freesolv": dict(name="FreeSolv", task="regression", num_classes=1),
    "drugbank": dict(name="DrugBank", task="classification", num_classes=86),
}


def quick(ds_key, ds):
    cfg = HCFLConfig()
    cfg.data.name = ds["name"]
    cfg.data.task = ds["task"]
    cfg.data.num_classes = ds["num_classes"]
    cfg.data.dirichlet_alpha = 0.5
    cfg.federated.num_clients = 50
    cfg.federated.participation_rate = 0.2
    cfg.federated.num_rounds = 50
    cfg.model.freeze_backbone_layers = 10
    cfg.contribution.estimator = "hybrid"
    cfg.save(os.path.join(HERE, f"{ds_key}.yaml"))


def paper(ds_key, ds):
    cfg = HCFLConfig()
    cfg.data.name = ds["name"]
    cfg.data.task = ds["task"]
    cfg.data.num_classes = ds["num_classes"]
    cfg.data.dirichlet_alpha = 0.5
    cfg.federated.num_clients = 1000
    cfg.federated.participation_rate = 0.03
    cfg.federated.num_rounds = 100
    cfg.model.freeze_backbone_layers = 0
    cfg.contribution.estimator = "hybrid"
    cfg.save(os.path.join(HERE, f"{ds_key}_paper.yaml"))


if __name__ == "__main__":
    for k, v in DATASETS.items():
        quick(k, v)
    paper("caco2", DATASETS["caco2"])
    print("Wrote configs:", sorted(f for f in os.listdir(HERE) if f.endswith(".yaml")))

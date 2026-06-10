# Dataset download/partition/persist and a shared DataContext for server+clients.
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from hcfl.config import HCFLConfig
from hcfl.data.download import load_tdc_dataset, inject_label_noise
from hcfl.data.partition import dirichlet_partition, assign_workloads, partition_summary
from hcfl.data.dataset import MoleculeDataset, build_collate_fn, smiles_list


def prepare_and_save(cfg: HCFLConfig) -> dict:
    dcfg = cfg.data
    out_dir = os.path.join(dcfg.data_root, dcfg.name)
    os.makedirs(out_dir, exist_ok=True)

    df, spec = load_tdc_dataset(dcfg.name, dcfg.data_root)
    rng = np.random.default_rng(dcfg.seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n_test = int(len(df) * dcfg.test_frac)
    n_val = int(len(df) * dcfg.val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    train_df = inject_label_noise(train_df, spec.task, dcfg.label_noise_std_frac, dcfg.seed)
    full = df.copy()
    full.loc[train_idx, "label"] = train_df["label"].to_numpy()

    local_parts = dirichlet_partition(
        full.iloc[train_idx].reset_index(drop=True),
        num_clients=cfg.federated.num_clients,
        alpha=dcfg.dirichlet_alpha,
        task=spec.task,
        seed=dcfg.seed,
    )
    train_global = np.asarray(train_idx)
    parts_global = [train_global[p] for p in local_parts]
    parts_global = assign_workloads(
        parts_global, dcfg.min_samples_per_client, dcfg.max_samples_per_client, seed=dcfg.seed
    )

    summary = partition_summary(parts_global, full, spec.task)

    full.to_pickle(os.path.join(out_dir, "df.pkl"))
    meta = {
        "spec": {
            "tdc_name": spec.tdc_name, "task": spec.task, "is_pair": spec.is_pair,
            "num_classes": spec.num_classes, "metric": spec.metric,
        },
        "test_idx": test_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "partitions": {str(k): p.tolist() for k, p in enumerate(parts_global)},
        "config": cfg.to_dict(),
        "summary": summary,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return {"out_dir": out_dir, **summary}


@dataclass
class DataContext:
    df: pd.DataFrame
    spec: object
    test_idx: np.ndarray
    val_idx: np.ndarray
    partitions: dict[int, np.ndarray]
    tokenizer: object
    cfg: HCFLConfig

    def _loader(self, indices, batch_size, shuffle):
        ds = MoleculeDataset(
            self.df, indices, self.tokenizer, self.spec.task, self.spec.is_pair,
            max_len=self.cfg.data.max_smiles_len,
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            collate_fn=build_collate_fn(self.spec.is_pair), num_workers=0,
        )

    def client_loader(self, indices, shuffle=True):
        return self._loader(indices, self.cfg.federated.batch_size, shuffle)

    def test_loader(self):
        return self._loader(self.test_idx, self.cfg.federated.batch_size, False)

    def reference_loader(self):
        n = min(len(self.val_idx),
                self.cfg.contribution.fim_batch_size * self.cfg.contribution.fim_max_batches * 4)
        return self._loader(self.val_idx[:n], self.cfg.contribution.fim_batch_size, False)

    def smiles_for(self, indices):
        return smiles_list(self.df, indices, self.spec.is_pair)


def load_context(cfg: HCFLConfig) -> DataContext:
    from types import SimpleNamespace

    from hcfl.models.chemberta import build_tokenizer

    out_dir = os.path.join(cfg.data.data_root, cfg.data.name)
    df = pd.read_pickle(os.path.join(out_dir, "df.pkl"))
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    spec = SimpleNamespace(**meta["spec"])
    partitions = {int(k): np.asarray(v, dtype=int) for k, v in meta["partitions"].items()}
    tokenizer = build_tokenizer(cfg.model.backbone_name)
    return DataContext(
        df=df, spec=spec,
        test_idx=np.asarray(meta["test_idx"], dtype=int),
        val_idx=np.asarray(meta["val_idx"], dtype=int),
        partitions=partitions, tokenizer=tokenizer, cfg=cfg,
    )

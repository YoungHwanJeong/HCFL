# HCFL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20623947.svg)](https://doi.org/10.5281/zenodo.20623947)

Reference code for **HCFL: Hybrid Contribution-Driven Federated Learning for Fair
and Efficient Optimization**. Built on [NVFlare](https://github.com/NVIDIA/NVFlare),
evaluated on molecular datasets from [PyTDC](https://tdcommons.ai/).

## Install

Python ≥ 3.10. Install PyTorch for your platform/CUDA first, then the package.

```bash
conda create -y -n hcfl python=3.11
conda activate hcfl

pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -e .
```

On aarch64, if PyTDC's `rdkit-pypi` dependency conflicts, install it without deps:
`pip install --no-deps "PyTDC==0.4.1"`.

## Run

Datasets: `Caco2_Wang`, `FreeSolv` (regression), `DrugBank` (86-class DDI).

```bash
# 1. download + Dirichlet non-IID partition
python scripts/prepare_data.py --config configs/caco2.yaml

# 2a. train (standalone simulator)
python scripts/run_simulation.py --config configs/caco2.yaml

# 2b. train (NVFlare)
python scripts/run_nvflare.py --config configs/caco2.yaml --threads 4 --gpu 0
```

Common overrides: `--dataset {Caco2_Wang,FreeSolv,DrugBank}`,
`--alpha {0.1,0.3,0.5,1.0}`, `--estimator {hybrid,individual,loo,shap,leastcore}`.
Results are written to `results/`.

`configs/*.yaml` are reduced-scale (single machine). `configs/*_paper.yaml` use the
paper setup (1000 clients, 3% participation, 100 rounds, full fine-tuning).

## Test

```bash
pytest tests/ -q
```

## Code availability

This repository contains the custom code implementing HCFL. The version used in
the paper is archived on Zenodo: https://doi.org/10.5281/zenodo.20623947

## Cite

```bibtex
@article{jeong2026hcfl,
  title  = {HCFL: Hybrid Contribution-Driven Federated Learning for Fair and Efficient Optimization},
  author = {Jeong, Younghwan and Lee, Sangshin and Lee, Jinyoung and Choi, Won Gi},
  year   = {2026}
}
```

MIT License.

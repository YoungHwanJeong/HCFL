# Coverage-based contribution C_k: sparse-region volume of client embeddings.
from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from scipy.stats import qmc


def robust_mean_cov(Z: np.ndarray, shrinkage: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
    d = Z.shape[1]
    mu = np.median(Z, axis=0)
    Zc = Z - mu
    n = max(Z.shape[0], 1)
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    trace_mean = np.trace(cov) / d if d > 0 else 1.0
    cov = (1 - shrinkage) * cov + shrinkage * trace_mean * np.eye(d)
    cov += 1e-6 * np.eye(d)
    return mu, cov


class GlobalDensity:
    def __init__(self, Z_G: np.ndarray, threshold_quantile: float = 0.30,
                 max_ref: int = 4000, seed: int = 42):
        rng = np.random.default_rng(seed)
        if Z_G.shape[0] > max_ref:
            idx = rng.choice(Z_G.shape[0], size=max_ref, replace=False)
            Z_G = Z_G[idx]
        self.Z_G = np.ascontiguousarray(Z_G, dtype=np.float64)
        self.eps_d = self._median_pairwise(self.Z_G, rng)
        ref_dens = self.density(self.Z_G)
        self.tau = float(np.quantile(ref_dens, threshold_quantile)) if len(ref_dens) else 0.0

    @staticmethod
    def _median_pairwise(Z: np.ndarray, rng, sample: int = 1000) -> float:
        n = Z.shape[0]
        if n < 2:
            return 1.0
        m = min(sample, n)
        a = Z[rng.choice(n, m, replace=False)]
        b = Z[rng.choice(n, m, replace=False)]
        dists = np.linalg.norm(a - b, axis=1)
        med = float(np.median(dists))
        return med if med > 0 else 1.0

    def density(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points)
        out = np.empty(points.shape[0], dtype=np.float64)
        n_g = self.Z_G.shape[0]
        chunk = 256
        for s in range(0, points.shape[0], chunk):
            p = points[s : s + chunk]
            d2 = (
                np.sum(p * p, axis=1, keepdims=True)
                - 2 * p @ self.Z_G.T
                + np.sum(self.Z_G * self.Z_G, axis=1)[None, :]
            )
            within = (d2 <= self.eps_d ** 2).sum(axis=1)
            out[s : s + chunk] = within / max(n_g, 1)
        return out


class CoverageContribution:
    def __init__(
        self,
        chi2_quantile: float = 0.95,
        sparsity_power: float = 2.0,
        qmc_samples: int = 512,
        cov_shrinkage: float = 1e-2,
        shape_penalty_weight: float = 0.10,
        vol_penalty_weight: float = 0.50,
        eps: float = 1e-8,
        seed: int = 42,
    ):
        self.chi2_quantile = chi2_quantile
        self.p = sparsity_power
        self.M = qmc_samples
        self.cov_shrinkage = cov_shrinkage
        self.eta_shape = shape_penalty_weight
        self.lambda_vol = vol_penalty_weight
        self.eps = eps
        self.seed = seed

    def _ellipsoid(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        d = Z.shape[1]
        mu, Sigma = robust_mean_cov(Z, self.cov_shrinkage)
        r2 = float(chi2.ppf(self.chi2_quantile, df=d))
        sign, logdet = np.linalg.slogdet(Sigma)
        from scipy.special import gammaln
        log_cd = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)
        log_vol = log_cd + 0.5 * logdet + d * 0.5 * np.log(r2)
        eigvals = np.linalg.eigvalsh(Sigma)
        cond = float(eigvals[-1] / max(eigvals[0], 1e-12))
        return mu, Sigma, r2, float(log_vol), cond

    def _sample_ellipsoid(self, mu, Sigma, r2, n, client_seed) -> tuple[np.ndarray, float]:
        d = mu.shape[0]
        L = np.linalg.cholesky(r2 * Sigma)
        sob = qmc.Sobol(d=d, scramble=True, seed=client_seed)
        m_pow = int(np.ceil(np.log2(max(n, 2))))
        u = sob.random_base2(m=m_pow)[:n]
        u = np.clip(u, 1e-9, 1 - 1e-9)
        from scipy.stats import norm
        g = norm.ppf(u)
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dirs = g / norms
        rad = np.power(np.random.default_rng(client_seed).random(n), 1.0 / d)[:, None]
        unit = dirs * rad
        pts = unit @ L.T + mu
        log_vol = self._log_vol(L, d)
        return pts, log_vol

    @staticmethod
    def _log_vol(L: np.ndarray, d: int) -> float:
        from scipy.special import gammaln
        log_cd = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)
        log_det_half = np.sum(np.log(np.abs(np.diag(L))))
        return log_cd + log_det_half

    def raw_scores(self, embeddings: dict[str, np.ndarray], density: GlobalDensity) -> dict:
        out = {}
        for i, (cid, Z) in enumerate(embeddings.items()):
            Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
            d = Z.shape[1]
            if Z.shape[0] <= d:
                out[cid] = self._fallback(cid, Z, density)
                continue
            mu, Sigma, r2, _, cond = self._ellipsoid(Z)
            pts, log_vol = self._sample_ellipsoid(mu, Sigma, r2, self.M, self.seed + i)
            rho = density.density(pts)
            mask = rho < density.tau
            if mask.any():
                weights = (density.tau / (rho[mask] + self.eps)) ** self.p
                mean_w = weights.sum() / self.M
            else:
                mean_w = 0.0
            c_hat = float(np.exp(log_vol) * mean_w) if mean_w > 0 else 0.0
            out[cid] = {"c_hat": c_hat, "log_vol": log_vol, "cond": cond}
        return out

    def _fallback(self, cid, Z, density: GlobalDensity) -> dict:
        if Z.shape[0] == 0:
            return {"c_hat": 0.0, "log_vol": -np.inf, "cond": 1.0}
        rho = density.density(Z)
        mask = rho < density.tau
        c_hat = float(((density.tau / (rho[mask] + self.eps)) ** self.p).mean()) if mask.any() else 0.0
        return {"c_hat": c_hat, "log_vol": 0.0, "cond": 1.0}

    def scores(self, embeddings: dict[str, np.ndarray], density: GlobalDensity) -> dict[str, float]:
        raw = self.raw_scores(embeddings, density)
        c_hats = np.array([v["c_hat"] for v in raw.values()], dtype=np.float64)
        mean_c = c_hats.mean() if len(c_hats) else 0.0
        log_vols = np.array([v["log_vol"] for v in raw.values() if np.isfinite(v["log_vol"])])
        mean_log_vol = log_vols.mean() if len(log_vols) else 0.0

        final = {}
        for cid, v in raw.items():
            c_norm = v["c_hat"] / (mean_c + self.eps)
            vol_ratio = np.exp(np.clip(v["log_vol"] - mean_log_vol, -50, 50))
            phi_shape = np.log(max(v["cond"], 1.0)) + self.lambda_vol * vol_ratio
            final[cid] = max(0.0, c_norm - self.eta_shape * phi_shape)
        return final

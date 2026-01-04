from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd

import hssm
from hssm.param.user_param import UserParam


@dataclass(frozen=True)
class FitConfig:
    sampler: str = "nuts_numpyro"
    draws: int = 500
    tune: int = 500
    chains: int = 2
    target_accept: float = 0.95
    random_seed: int = 1
    progress_bar: bool = False


@dataclass(frozen=True)
class FitResult:
    idata: az.InferenceData
    loo: az.ELPDData | None
    beta_summary: dict[str, float] | None


def _beta_summary(idata: az.InferenceData, var: str) -> dict[str, float] | None:
    if var not in idata.posterior:
        return None
    x = idata.posterior[var].values.reshape(-1)
    hdi = az.hdi(x, hdi_prob=0.95)
    return {
        "mean": float(np.mean(x)),
        "hdi_2.5": float(hdi[0]),
        "hdi_97.5": float(hdi[1]),
        "p_gt_0": float(np.mean(x > 0)),
    }


def fit_ddm(
    data: pd.DataFrame,
    *,
    v_formula: str,
    group: str = "subject",
    fit_cfg: FitConfig,
    loglik_kind: str = "analytical",
    z_fixed: float = 0.5,
    beta_varname: str | None = None,
    hierarchical_v_only: bool = True,
) -> FitResult:
    if len(data) == 0:
        raise ValueError("Empty dataset for fitting.")
    needed = {"rt", "response", group}
    if not needed.issubset(set(data.columns)):
        raise ValueError(f"Missing required columns: {sorted(needed - set(data.columns))}")

    if hierarchical_v_only:
        include = [
            UserParam("v", formula=v_formula),
            UserParam("a", formula="a ~ 1"),
            UserParam("t", formula="t ~ 1"),
            UserParam("z", prior=float(z_fixed)),
        ]
    else:
        include = [
            UserParam("v", formula=v_formula),
            UserParam("a", formula=f"a ~ 1 + (1|{group})"),
            UserParam("t", formula=f"t ~ 1 + (1|{group})"),
            UserParam("z", prior=float(z_fixed)),
        ]

    model = hssm.HSSM(
        data,
        model="ddm",
        loglik_kind=loglik_kind,
        include=include,
    )
    idata = model.sample(
        sampler=fit_cfg.sampler,
        draws=int(fit_cfg.draws),
        tune=int(fit_cfg.tune),
        chains=int(fit_cfg.chains),
        target_accept=float(fit_cfg.target_accept),
        random_seed=int(fit_cfg.random_seed),
        progress_bar=bool(fit_cfg.progress_bar),
    )
    beta = _beta_summary(idata, beta_varname) if beta_varname else None
    loo = None
    try:
        loo = az.loo(idata)
    except Exception:
        loo = None
    return FitResult(idata=idata, loo=loo, beta_summary=beta)

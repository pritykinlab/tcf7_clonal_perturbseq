"""
Isotonic regression helpers reused across manuscript scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def _most_common(series: pd.Series) -> str:
    mode = series.mode()
    if not mode.empty:
        return mode.iloc[0]
    return series.iloc[0]


def _sanitize_cell_type(name: str) -> str:
    return name.lower().replace(' ', '_').replace('-', '_')


def _clone_metadata_tables(adata) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obs = adata.obs[['clone', 'cell_type', 'perturbation', 'guide', 'condition']].copy()
    obs = obs.astype(str)

    metadata = obs.groupby('clone')[['perturbation', 'guide', 'condition']].agg(_most_common)
    num_cells = obs.groupby('clone').size().rename('num_cells')

    counts = (
        obs.assign(_count=1)
        .pivot_table(index='clone', columns='cell_type', values='_count', aggfunc='sum', fill_value=0)
    )

    base = pd.concat([num_cells, metadata], axis=1)
    return base, counts


def prepare_clone_counts_df(adata, cell_types: Sequence[str]) -> pd.DataFrame:
    """Return clone-level DataFrame with counts for each requested cell type."""

    base, counts = _clone_metadata_tables(adata)
    counts = counts.reindex(columns=cell_types, fill_value=0)

    df = base.copy()
    for cell_type in cell_types:
        col_name = f"{_sanitize_cell_type(cell_type)}_count"
        df[col_name] = counts[cell_type].astype(int)

    df.reset_index(inplace=True)
    return df


def prepare_clone_proportions_df(adata, cell_types: Sequence[str]) -> pd.DataFrame:
    """Return clone-level DataFrame with proportions for each requested cell type."""

    base, counts = _clone_metadata_tables(adata)
    counts = counts.reindex(columns=cell_types, fill_value=0)
    proportions = counts.div(base['num_cells'], axis=0).fillna(0)

    df = base.join(proportions)
    df.reset_index(inplace=True)
    return df


@dataclass
class IsotonicCIResult:
    """Container holding isotonic predictions and bootstrap confidence intervals."""

    x_grid: np.ndarray
    fit: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    bootstrap_mean: np.ndarray
    bootstrap_samples: int
    direction: str


def _build_iso(increasing: bool) -> IsotonicRegression:
    """Return IsotonicRegression with consistent out-of-bounds handling."""
    return IsotonicRegression(increasing=increasing, out_of_bounds="clip")


def _choose_direction(x: np.ndarray, y: np.ndarray) -> Tuple[IsotonicRegression, bool]:
    """Select monotonic direction by comparing mean squared error."""

    iso_inc = _build_iso(True).fit(x, y)
    iso_dec = _build_iso(False).fit(x, y)

    mse_inc = np.mean((y - iso_inc.predict(x)) ** 2)
    mse_dec = np.mean((y - iso_dec.predict(x)) ** 2)

    if mse_inc <= mse_dec:
        return iso_inc, True
    return iso_dec, False


def isotonic_with_ci(
    x: Iterable[float],
    y: Iterable[float],
    *,
    x_grid: Optional[np.ndarray] = None,
    direction: str = "auto",
    n_bootstrap: int = 500,
    random_state: Optional[int] = 42,
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> IsotonicCIResult:
    """Fit isotonic regression and compute bootstrap-based confidence intervals."""

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.all(valid_mask):
        x_arr = x_arr[valid_mask]
        y_arr = y_arr[valid_mask]

    if x_arr.size == 0:
        raise ValueError("Input arrays contain no valid observations for isotonic regression.")

    if x_grid is None:
        x_grid = np.linspace(x_arr.min(), x_arr.max(), 100)

    direction_key = direction.lower()
    if direction_key not in {"auto", "increasing", "decreasing"}:
        raise ValueError(f"Unsupported direction '{direction}'. Use 'auto', 'increasing', or 'decreasing'.")

    if direction_key == "increasing":
        iso_model = _build_iso(True).fit(x_arr, y_arr)
        is_increasing = True
    elif direction_key == "decreasing":
        iso_model = _build_iso(False).fit(x_arr, y_arr)
        is_increasing = False
    else:
        iso_model, is_increasing = _choose_direction(x_arr, y_arr)

    fit = iso_model.predict(x_grid)

    if transform is not None:
        fit = transform(fit)

    rng = np.random.default_rng(random_state)
    bootstrap_predictions = []

    for _ in range(max(n_bootstrap, 0)):
        sample_idx = rng.integers(0, x_arr.size, size=x_arr.size)
        x_sample = x_arr[sample_idx]
        y_sample = y_arr[sample_idx]

        try:
            iso_boot = _build_iso(is_increasing).fit(x_sample, y_sample)
            boot_pred = iso_boot.predict(x_grid)
            if transform is not None:
                boot_pred = transform(boot_pred)
            bootstrap_predictions.append(boot_pred)
        except ValueError:
            continue

    if bootstrap_predictions:
        stacked = np.vstack(bootstrap_predictions)
        lower = np.percentile(stacked, 2.5, axis=0)
        upper = np.percentile(stacked, 97.5, axis=0)
        mean = stacked.mean(axis=0)
        n_successful = stacked.shape[0]
    else:
        lower = upper = mean = np.full_like(fit, np.nan)
        n_successful = 0

    return IsotonicCIResult(
        x_grid=x_grid,
        fit=fit,
        ci_lower=lower,
        ci_upper=upper,
        bootstrap_mean=mean,
        bootstrap_samples=n_successful,
        direction="increasing" if is_increasing else "decreasing",
    )

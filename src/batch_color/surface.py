"""Conservative spatial support gates and blocked validation for background fits.

Thresholds are engineering guardrails, not calibrated perceptual quality scores.
"""
import numpy as np


def support_is_valid(d):
    required = {"x_span": (1.2, 2.01), "y_span": (1.2, 2.01),
                "grid_coverage": (0.25, 1), "quadrants": (4, 4),
                "rank": (6, 6), "condition_number": (1, 1000),
                "max_grid_distance": (0, 0.9), "far_area_fraction": (0, 0.15)}
    return all(isinstance(d.get(k), (int, float)) and not isinstance(d[k], bool)
               and np.isfinite(d[k]) and low <= d[k] <= high
               for k, (low, high) in required.items())


def support_diagnostics(features, x, y):
    cells_x = np.clip(((x + 1) * 4).astype(int), 0, 7)
    cells_y = np.clip(((y + 1) * 4).astype(int), 0, 7)
    cells = np.unique(cells_y * 8 + cells_x)
    centers = np.stack((cells % 8, cells // 8), axis=-1) / 4 - 0.875
    grid_y, grid_x = np.mgrid[:8, :8]
    grid = np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1) / 4 - 0.875
    distances = np.linalg.norm(grid[:, None] - centers[None, :], axis=-1).min(axis=1)
    condition = float(np.linalg.cond(features.astype(np.float64)))
    d = {"x_span": float(np.ptp(x)), "y_span": float(np.ptp(y)),
         "grid_coverage": float(len(cells) / 64),
         "quadrants": int(len(np.unique((x >= 0).astype(int) + 2 * (y >= 0).astype(int)))),
         "rank": int(np.linalg.matrix_rank(features.astype(np.float64))),
         "condition_number": min(condition, 1e30) if np.isfinite(condition) else 1e30,
         "max_grid_distance": float(distances.max()),
         "far_area_fraction": float(np.mean(distances > 0.75))}
    # Spatial blocks, not randomly interleaved pixels in the same narrow strip.
    folds = (cells_x // 2 + 2 * (cells_y // 2)) % 4
    return d, folds


def _fit(features, values, size):
    f = features[:, :size].astype(np.float64)
    v = values.astype(np.float64)
    if size == 1:
        return np.median(v, axis=0)[None, :]
    c, *_ = np.linalg.lstsq(f, v, rcond=None)
    for _ in range(3):
        errors = np.linalg.norm(v - f @ c, axis=1)
        center = np.median(errors)
        scale = max(float(np.median(np.abs(errors - center))) * 1.4826, 0.006)
        weight = np.sqrt(1 / (1 + np.maximum((errors - center) / (3 * scale), 0) ** 2))
        c, *_ = np.linalg.lstsq(f * weight[:, None], v * weight[:, None], rcond=None)
    return c


def choose_surface(features, values, x, y, bounds_check):
    diagnostics, folds = support_diagnostics(features, x, y)
    model, chosen_size = "constant", 1
    trusted = support_is_valid(diagnostics)
    diagnostics["reason"] = "support_passed" if trusted else "insufficient_spatial_support"
    errors = {}
    if trusted:
        for name, size in (("constant", 1), ("plane", 3), ("quadratic", 6)):
            held_errors = []
            for fold in range(4):
                train, valid = folds != fold, folds == fold
                if valid.sum() < 30 or train.sum() < 100:
                    trusted = False
                    break
                f = features[train, :size].astype(np.float64)
                if np.linalg.matrix_rank(f) < size or np.linalg.cond(f) > 1000:
                    trusted = False
                    break
                c = _fit(features[train], values[train], size)
                held_errors.append(float(np.sqrt(np.mean(np.sum(
                    (values[valid] - features[valid, :size] @ c) ** 2, axis=1)))))
            if not trusted:
                diagnostics["reason"] = "invalid_spatial_holdout"
                break
            errors[name] = float(np.mean(held_errors))
            if name != "constant" and errors[name] < errors[model] - max(0.001, 0.10 * errors[model]):
                model, chosen_size = name, size
        if trusted and errors[model] > 0.05:
            trusted = False
            diagnostics["reason"] = "holdout_error_too_large"
    if not trusted:
        model, chosen_size = "constant", 1
    coefficients = np.zeros((3, 6), dtype=np.float64)
    coefficients[:, :chosen_size] = _fit(features, values, chosen_size).T
    if not bounds_check(coefficients):
        trusted, model, chosen_size = False, "constant", 1
        diagnostics["reason"] = "surface_color_or_gradient_bounds"
        coefficients[:] = 0
        coefficients[:, :1] = _fit(features, values, 1).T
    diagnostics["blocked_validation_rmse"] = errors
    diagnostics["selected_model"] = model
    residual = float(np.median(np.linalg.norm(values - features @ coefficients.T, axis=1)))
    return coefficients, residual, model, trusted, diagnostics

"""Fit once on A1 reference records, apply frozen utilities to all sources."""
import copy
import numpy as np
import pandas as pd
from .schema import FIELDS, RAW_COLUMNS


def balanced_weights(domains):
    domains = np.asarray(domains)
    w = np.zeros(len(domains))
    groups, counts = np.unique(domains, return_counts=True)
    for g, n in zip(groups, counts):
        w[domains == g] = 1 / (len(groups) * n)
    return w


def weighted_quantile(x, q, weights=None):
    x = np.asarray(x, float)
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    keep = np.isfinite(x) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return np.full(np.asarray(q).shape, np.nan)
    x, weights = x[keep], weights[keep]
    order = np.argsort(x, kind="stable")
    x, weights = x[order], weights[order]
    positions = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(q, positions, x)


def validate_config(config):
    if set(config["utilities"]) != set(FIELDS):
        raise ValueError("Utilities must cover exactly all 22 quality fields")
    lo, hi = config["conflict"]["low"], config["conflict"]["high"]
    if not 0 < lo < hi < 1:
        raise ValueError("Conflict thresholds require 0 < low < high < 1")
    if not 0 <= config["conflict"]["eta"] <= 1:
        raise ValueError("eta must lie in [0,1]")
    if not 0 <= config["weights"]["critic_shrinkage"] <= 1:
        raise ValueError("CRITIC shrinkage must lie in [0,1]")
    q = np.asarray(config["qurater_weights"], float)
    if q.shape != (4,) or (q < 0).any() or not np.isclose(q.sum(), 1):
        raise ValueError("Four QuRating weights must be nonnegative and sum to one")
    if not 0 < config["reference"]["evaluation_fraction"] < 1:
        raise ValueError("A1 evaluation fraction must lie in (0,1)")
    if config.get("primary_score", "Q_base") not in {"Q_base", "Q_rule"}:
        raise ValueError("primary_score must be Q_base or Q_rule")


def fit_preprocessor(raw, domains, config):
    domains = np.asarray(domains)
    weights = balanced_weights(domains) if config["reference"]["domain_balanced"] else np.ones(len(domains)) / len(domains)
    global_medians = [float(weighted_quantile(raw[:, j], .5, weights)) for j in range(raw.shape[1])]
    unavailable = [RAW_COLUMNS[j] for j, m in enumerate(global_medians) if not np.isfinite(m)]
    global_medians = [0.0 if not np.isfinite(m) else m for m in global_medians]
    medians = {}
    for g in np.unique(domains):
        medians[str(g)] = [float(weighted_quantile(raw[domains == g, j], .5)) if np.isfinite(raw[domains == g, j]).any() else global_medians[j]
                           for j in range(raw.shape[1])]
    specs = {}
    for j, column in enumerate(RAW_COLUMNS):
        field = "qurater" if column.startswith("qurater_") else column
        spec = copy.deepcopy(config["utilities"][field])
        if field == "qurater":
            spec["kind"] = "reference_positive"
        x = raw[:, j].copy()
        if spec.get("log1p"):
            x = np.log1p(np.maximum(x, 0))
        if spec["kind"] in ("reference_positive", "reference_negative"):
            bounds = weighted_quantile(x, config["reference"]["quantiles"], weights)
            spec["bounds"] = [float(v) if np.isfinite(v) else 0.0 for v in bounds]
        elif spec["kind"] == "reference_interval":
            bounds = weighted_quantile(x, spec["interval_quantiles"], weights)
            spec["bounds"] = [float(v) if np.isfinite(v) else 0.0 for v in bounds]
            spec["scale"] = max((spec["bounds"][1] - spec["bounds"][0]) / 2, 1e-6)
        specs[column] = spec
    return {"version": config["version"], "reference_count": len(raw), "domain_counts": pd.Series(domains).value_counts().to_dict(),
            "global_medians": global_medians, "domain_medians": medians, "specs": specs,
            "unavailable_reference_columns": unavailable, "qurater_weights": config["qurater_weights"],
            "scale_status": "fixed_preferences_and_empirical_reference_not_human_calibrated"}


def transform(raw, domains, fitted, domain_allowance=True):
    domains = np.asarray(domains)
    filled = raw.copy()
    valid_raw = np.isfinite(raw)
    for g in np.unique(domains):
        mask = domains == g
        med = np.asarray(fitted["domain_medians"].get(str(g), fitted["global_medians"]))
        filled[mask] = np.where(valid_raw[mask], filled[mask], med)
    utilities, applicable = [], []
    for j, column in enumerate(RAW_COLUMNS):
        spec = fitted["specs"][column]
        x = filled[:, j]
        if spec.get("log1p"):
            x = np.log1p(np.maximum(x, 0))
        kind = spec["kind"]
        if kind == "identity":
            z = np.clip(x, 0, 1)
        elif kind in ("fixed_positive", "reference_positive", "reference_negative"):
            lower, upper = spec["bounds"]
            z = np.full(len(x), .5) if upper-lower < 1e-12 else np.clip((x-lower)/(upper-lower), 0, 1)
            if kind == "reference_negative":
                z = 1-z
        elif kind == "reference_interval":
            lower, upper = spec["bounds"]
            distance = np.maximum(lower-x, 0) + np.maximum(x-upper, 0)
            z = np.exp(-(distance/spec["scale"])**2)
        elif kind == "upper_tolerance":
            tolerance = np.full(len(x), spec["tolerance"], float)
            if domain_allowance:
                for g, value in spec.get("domain_tolerance", {}).items():
                    tolerance[domains == g] = value
            scale = np.maximum((1-tolerance)/2, 1e-6)
            z = np.exp(-(np.maximum(x-tolerance, 0)/scale)**2)
        else:
            raise ValueError(f"Unknown utility kind: {kind}")
        app = np.ones(len(x), bool)
        if domain_allowance:
            for g in spec.get("neutral_domains", []):
                app[domains == g] = False
                z[domains == g] = .5
        if column in fitted["unavailable_reference_columns"]:
            z[:] = .5
        utilities.append(z)
        applicable.append(app)
    u = np.column_stack(utilities)
    app_raw = np.column_stack(applicable)
    z, valid, app = [], [], []
    for field in FIELDS:
        if field == "qurater":
            js = [RAW_COLUMNS.index(f"qurater_{i}") for i in range(4)]
            z.append(u[:, js] @ np.asarray(fitted["qurater_weights"]))
            valid.append(valid_raw[:, js].all(axis=1))
            app.append(app_raw[:, js].all(axis=1))
        else:
            j = RAW_COLUMNS.index(field)
            z.append(u[:, j]); valid.append(valid_raw[:, j]); app.append(app_raw[:, j])
    return np.column_stack(z), np.column_stack(valid), np.column_stack(app), u


def utility_dictionary(fitted):
    rows = []
    for c in RAW_COLUMNS:
        s = fitted["specs"][c]
        rows.append({"column": c, "kind": s["kind"], "bounds": str(s.get("bounds", "")),
                     "scale": s.get("scale", ""), "basis": s["basis"],
                     "domain_tolerance": str(s.get("domain_tolerance", {})),
                     "neutral_domains": str(s.get("neutral_domains", [])),
                     "log1p": s.get("log1p", False)})
    return pd.DataFrame(rows)

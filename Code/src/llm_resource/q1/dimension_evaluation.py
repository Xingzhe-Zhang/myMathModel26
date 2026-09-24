"""Descriptive comparisons and paired evaluation against independent labels."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_correlation(x, y) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def top_mask(values, fraction) -> np.ndarray:
    values = np.asarray(values, float)
    selected = np.zeros(len(values), bool)
    if len(values):
        count = max(1, int(np.ceil(fraction * len(values))))
        selected[np.argsort(-values, kind="stable")[:count]] = True
    return selected


def ks_distance(x, y) -> float:
    """Two-sample KS distance, used descriptively rather than as a quality test."""
    x, y = np.sort(np.asarray(x, float)), np.sort(np.asarray(y, float))
    if not len(x) or not len(y):
        return float("nan")
    grid = np.sort(np.concatenate([x, y]))
    return float(np.max(np.abs(np.searchsorted(x, grid, side="right") / len(x)
                               - np.searchsorted(y, grid, side="right") / len(y))))


def summarize_unlabeled(scores: pd.DataFrame, masks: dict, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    domains = scores["domain"].to_numpy()
    rows, comparisons, shifts = [], [], []
    for source in ("A1", "A2", "A3", "A2_new", "A3_new", "union"):
        selected = masks[source]
        for domain in np.unique(domains[selected]):
            mask = selected & (domains == domain)
            x = scores.loc[mask, "Q22_equal"].to_numpy()
            y = scores.loc[mask, "Q_semantic"].to_numpy()
            overlap = (top_mask(x, fraction) & top_mask(y, fraction)).sum() / max(top_mask(x, fraction).sum(), 1)
            rows.append({"source": source, "domain": domain, "n_unique": len(x),
                         "Q22_mean": x.mean(), "Q_semantic_mean": y.mean(),
                         "Q22_std": x.std(), "Q_semantic_std": y.std(),
                         "rank_correlation": rank_correlation(x, y),
                         "top_fraction_overlap": overlap})
            if source == "A1":
                comparisons.append({"domain": domain, "n": len(x), "mean_change": (y-x).mean(),
                                    "mean_absolute_change": np.abs(y-x).mean(),
                                    "rank_correlation": rank_correlation(x, y),
                                    "top_fraction_overlap": overlap})
    summary = pd.DataFrame(rows)
    for domain, source in (("arxiv", "A2_new"), ("github", "A3_new")):
        a = masks["A1"] & (domains == domain)
        b = masks[source] & (domains == domain)
        for scheme in ("Q22_equal", "Q_semantic"):
            x = scores.loc[a, scheme].to_numpy()
            y = scores.loc[b, scheme].to_numpy()
            shifts.append({"domain": domain, "new_source": source, "scheme": scheme,
                           "n_A1": len(x), "n_new": len(y), "mean_A1": x.mean(),
                           "mean_new": y.mean(), "mean_shift": y.mean()-x.mean(),
                           "shift_in_A1_standard_deviations": (y.mean()-x.mean()) / max(x.std(), 1e-12),
                           "ks_distance": ks_distance(x, y),
                           "interpretation": "distribution shift only; the extension has no quality labels"})
    return summary, pd.DataFrame(comparisons), pd.DataFrame(shifts)


def _selected_metrics(frame: pd.DataFrame, fraction: float) -> tuple[dict, pd.DataFrame]:
    detail = []
    for domain, group in frame.groupby("domain", sort=True):
        for scheme in ("Q22_equal", "Q_semantic"):
            chosen = group.iloc[np.flatnonzero(top_mask(group[scheme].to_numpy(), fraction))]
            detail.append({"domain": domain, "scheme": scheme, "n": len(group),
                           "selected": len(chosen), "quality_selected": chosen["quality"].mean(),
                           "defect_rate_selected": chosen["defect"].mean()})
    detail = pd.DataFrame(detail)
    macro = {}
    for scheme in ("Q22_equal", "Q_semantic"):
        subset = detail[detail.scheme == scheme]
        macro[scheme] = {"macro_quality_selected": float(subset.quality_selected.mean()),
                         "macro_defect_rate_selected": float(subset.defect_rate_selected.mean())}
    return macro, detail


def evaluate_independent_labels(scores: pd.DataFrame, labels: pd.DataFrame, config: dict, seed: int) -> tuple[dict, pd.DataFrame]:
    required = {"uid", "quality", "defect"}
    if not required <= set(labels):
        raise ValueError("Labels require uid, quality (0..1), and defect (0/1) columns")
    if labels.uid.duplicated().any():
        raise ValueError("Labels contain duplicate uid values; adjudicate reviewers first")
    labels = labels.copy()
    labels["quality"] = pd.to_numeric(labels.quality, errors="raise")
    labels["defect"] = pd.to_numeric(labels.defect, errors="raise")
    if not labels.quality.between(0, 1).all() or not labels.defect.isin([0, 1]).all():
        raise ValueError("quality must be in [0,1] and defect must be 0 or 1")
    agreement = None
    reviewer_columns = {"quality_r1", "quality_r2", "defect_r1", "defect_r2"}
    if reviewer_columns <= set(labels):
        for column in reviewer_columns:
            labels[column] = pd.to_numeric(labels[column], errors="raise")
        if not labels.quality_r1.between(0, 1).all() or not labels.quality_r2.between(0, 1).all() or not labels.defect_r1.isin([0, 1]).all() or not labels.defect_r2.isin([0, 1]).all():
            raise ValueError("Reviewer labels must use the same quality and defect scales")
        observed = float((labels.defect_r1 == labels.defect_r2).mean())
        expected = float(labels.defect_r1.mean() * labels.defect_r2.mean()
                         + (1-labels.defect_r1.mean()) * (1-labels.defect_r2.mean()))
        agreement = {"defect_agreement": observed,
                     "defect_cohen_kappa": (observed-expected)/(1-expected) if expected < 1 else None,
                     "quality_mean_absolute_difference": float(np.abs(labels.quality_r1-labels.quality_r2).mean()),
                     "note": "Descriptive reviewer reliability; disagreements require independent adjudication."}
    reference = scores.loc[scores.is_A1_reference, ["uid", "domain", "Q22_equal", "Q_semantic"]]
    frame = reference.merge(labels[list(required)], on="uid", how="inner", validate="one_to_one")
    if len(frame) != len(labels):
        raise ValueError("Some label uid values do not belong to A1")
    settings = config["human_evaluation"]
    fraction = float(settings["selection_fraction_per_domain"])
    metrics, detail = _selected_metrics(frame, fraction)
    counts = frame.domain.value_counts()
    all_domains = set(reference.domain.unique())
    adequate = len(frame) >= settings["minimum_labeled_total"] and all(
        counts.get(domain, 0) >= settings["minimum_labeled_per_domain"] for domain in all_domains
    )
    result = {"n_labeled": len(frame), "domain_counts": counts.to_dict(),
              "selection_fraction_per_domain": fraction, "metrics": metrics,
              "status": "descriptive_only_insufficient_labels" if not adequate else "evaluated",
              "decision": "insufficient_evidence"}
    if agreement is not None:
        result["reviewer_agreement"] = agreement
    if not adequate:
        return result, detail
    rng = np.random.default_rng(seed)
    groups = [group.reset_index(drop=True) for _, group in frame.groupby("domain", sort=True)]
    diffs = []
    for _ in range(int(settings["bootstrap_repeats"])):
        boot = pd.concat([group.iloc[rng.integers(0, len(group), len(group))] for group in groups], ignore_index=True)
        m, _ = _selected_metrics(boot, fraction)
        diffs.append([m["Q_semantic"]["macro_quality_selected"] - m["Q22_equal"]["macro_quality_selected"],
                      m["Q_semantic"]["macro_defect_rate_selected"] - m["Q22_equal"]["macro_defect_rate_selected"]])
    diffs = np.asarray(diffs)
    quality_ci = np.quantile(diffs[:, 0], [0.025, 0.975]).tolist()
    defect_ci = np.quantile(diffs[:, 1], [0.025, 0.975]).tolist()
    result["paired_bootstrap_ci95"] = {"quality_difference_semantic_minus_baseline": quality_ci,
                                       "defect_difference_semantic_minus_baseline": defect_ci}
    quality_by_domain = detail.pivot(index="domain", columns="scheme", values="quality_selected")
    domain_harm = (quality_by_domain["Q_semantic"] - quality_by_domain["Q22_equal"]).min()
    if quality_ci[0] > 0 and defect_ci[1] < settings["max_defect_rate_increase"] and domain_harm > -settings["max_domain_quality_drop"]:
        result["decision"] = "semantic_supported_on_this_independent_label_set"
    elif quality_ci[1] < 0 or defect_ci[0] > settings["max_defect_rate_increase"]:
        result["decision"] = "baseline_supported_on_this_independent_label_set"
    else:
        result["decision"] = "inconclusive_collect_more_labels_or_proxy_training_feedback"
    result["rule"] = "Semantic needs positive lower 95% CI for selected quality, defect upper CI below configured noninferiority margin, and no large domain quality drop."
    return result, detail

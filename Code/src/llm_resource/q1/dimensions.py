"""Versioned 22-field baseline and 25-atomic-signal semantic representation.

The semantic view separates intrinsic quality candidates from complexity,
structure, and target relevance. Its core gives equal budgets to ten non-
QuRating concepts and preserves the original QuRating parent budget. These
are explicit preferences, not learned training-utility weights.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .schema import FIELDS, RAW_COLUMNS

QURATER_NAMES = (
    "qurater_writing_style",
    "qurater_required_expertise",
    "qurater_facts_trivia",
    "qurater_educational_value",
)
ATOMIC_FIELDS = tuple(
    QURATER_NAMES[int(name.rsplit("_", 1)[1])] if name.startswith("qurater_") else name
    for name in RAW_COLUMNS
)


def validate_dimension_config(config: dict) -> None:
    groups = config["groups"]
    if not groups or any(not concepts for concepts in groups.values()):
        raise ValueError("Each dimension group must contain concepts")
    flattened = [field for concepts in groups.values() for members in concepts for field in members]
    counts = Counter(flattened)
    if set(flattened) != set(ATOMIC_FIELDS) or any(count != 1 for count in counts.values()):
        raise ValueError("Every atomic field must occur exactly once across groups")
    if any(not members for concepts in groups.values() for members in concepts):
        raise ValueError("Empty concept")
    weights = config["quality_group_weights"]
    if not weights or set(weights) - set(groups):
        raise ValueError("Quality weights must name existing groups")
    values = np.asarray(list(weights.values()), float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1):
        raise ValueError("Quality group weights must be nonnegative and sum to one")
    q_budget = config["qurater_parent_budget"]
    if not np.isfinite(q_budget) or not 0 <= q_budget <= 1:
        raise ValueError("QuRating parent budget must be in [0,1]")
    quality_q = [field for group in weights for members in groups[group] for field in members if field in QURATER_NAMES]
    if len(quality_q) < 1 or len(quality_q) == len(QURATER_NAMES):
        raise ValueError("Quality view must contain some QuRating dimensions and leave complexity separate")
    for group in weights:
        if not any(all(field not in QURATER_NAMES for field in members) for members in groups[group]):
            raise ValueError("Each quality group needs a non-QuRating concept")
        if any(any(field in QURATER_NAMES for field in members) and
               any(field not in QURATER_NAMES for field in members) for members in groups[group]):
            raise ValueError("Do not mix QuRating and other fields within one concept")


def score_dimension_views(z22: np.ndarray, z25: np.ndarray, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_dimension_config(config)
    if z22.shape[1] != len(FIELDS) or z25.shape[1] != len(ATOMIC_FIELDS) or len(z22) != len(z25):
        raise ValueError("Expected aligned 22-field and 25-atomic utility matrices")
    atomic = {name: z25[:, j] for j, name in enumerate(ATOMIC_FIELDS)}
    scores = {"Q22_equal": z22.mean(axis=1)}
    effective = dict.fromkeys(ATOMIC_FIELDS, 0.0)
    for group, concepts in config["groups"].items():
        concept_values = [np.mean([atomic[field] for field in members], axis=0) for members in concepts]
        scores[f"G_{group}"] = np.mean(concept_values, axis=0)
        if group in config["quality_group_weights"]:
            nonq_concepts = [members for members in concepts if not any(field in QURATER_NAMES for field in members)]
            group_weight = (1-config["qurater_parent_budget"]) * config["quality_group_weights"][group]
            for members in nonq_concepts:
                for field in members:
                    effective[field] += group_weight / (len(nonq_concepts) * len(members))
    quality_q = [field for group in config["quality_group_weights"]
                 for members in config["groups"][group] for field in members if field in QURATER_NAMES]
    for field in quality_q:
        effective[field] = config["qurater_parent_budget"] / len(quality_q)
    scores["Q_semantic"] = sum(effective[field] * atomic[field] for field in ATOMIC_FIELDS)
    for field in QURATER_NAMES:
        scores[field] = atomic[field]
    baseline = {
        field: (1 / 88 if field in QURATER_NAMES else 1 / 22) for field in ATOMIC_FIELDS
    }
    weight_table = pd.DataFrame({
        "atomic_field": ATOMIC_FIELDS,
        "Q22_equal_weight": [baseline[field] for field in ATOMIC_FIELDS],
        "Q_semantic_weight": [effective[field] for field in ATOMIC_FIELDS],
        "semantic_group": [group for field in ATOMIC_FIELDS for group, concepts in config["groups"].items()
                           if any(field in members for members in concepts)],
    })
    return pd.DataFrame(scores), weight_table

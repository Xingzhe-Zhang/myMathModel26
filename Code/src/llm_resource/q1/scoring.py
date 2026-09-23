"""Monotone finite-compensation scores and preference sensitivity scenarios."""
import numpy as np
from .schema import FIELDS, MODERN


def critic_weights(z, row_weights):
    row_weights = np.asarray(row_weights, float)
    row_weights = row_weights / row_weights.sum()
    mean = row_weights @ z
    centered = z-mean
    cov = (centered * row_weights[:, None]).T @ centered
    std = np.sqrt(np.maximum(np.diag(cov), 0))
    denom = std[:, None]*std[None, :]
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-14)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1)
    info = std * (1-corr).sum(axis=1)
    return info/info.sum() if info.sum() > 0 else np.ones(z.shape[1])/z.shape[1]


def pair_indices(pairs):
    edges = [tuple(sorted((FIELDS.index(a), FIELDS.index(b)))) for a, b in pairs]
    if len(edges) != len(set(edges)) or any(a == b for a, b in edges):
        raise ValueError("Interaction edges must be unique undirected pairs")
    return edges


def interaction_weights(weights, edges, eta):
    weights = np.asarray(weights, float)
    if (weights < 0).any() or not np.isclose(weights.sum(), 1) or not 0 <= eta <= 1:
        raise ValueError("Invalid score weights or eta")
    degree = np.zeros(len(weights), int)
    for a, b in edges:
        degree[a] += 1; degree[b] += 1
    theta = np.asarray([eta*min(weights[a]/degree[a], weights[b]/degree[b]) for a,b in edges])
    validate_interactions(weights, edges, theta)
    return theta


def validate_interactions(weights, edges, theta):
    if len(edges) != len(theta) or np.any(np.asarray(theta) < 0):
        raise ValueError("Invalid interaction parameters")
    incidence = np.zeros(len(weights))
    for (a,b), t in zip(edges,theta):
        incidence[a] += t; incidence[b] += t
    if np.any(incidence-np.asarray(weights) > 1e-12):
        raise ValueError("Interaction parameters violate coordinate monotonicity")
    return incidence


def quality(z, weights, edges=(), theta=()):
    validate_interactions(weights, edges, theta)
    q0 = z @ np.asarray(weights)
    penalty = np.zeros(len(z))
    for (a,b),t in zip(edges,theta):
        penalty += t*np.abs(z[:,a]-z[:,b])
    return q0, q0-penalty, penalty


def huber_center(z, delta=.15):
    """Robust location across criteria; baseline only, not defect detection."""
    m = np.median(z, axis=1)
    for _ in range(30):
        residual = z-m[:,None]
        w = np.minimum(1, delta/np.maximum(np.abs(residual),1e-12))
        update = (w*z).sum(axis=1)/w.sum(axis=1)
        if np.max(np.abs(update-m)) < 1e-8:
            break
        m = update
    return update


def diagnostic_scores(z, weights, high, low):
    q0 = z @ weights
    dispersion = np.sqrt(np.maximum((z*z) @ weights-q0*q0,0))
    h = np.maximum(z-high,0)/(1-high)
    l = np.maximum(low-z,0)/low
    # High and low cannot occur in the same component, so this equals the
    # sum of all unordered-pair bidirectional products without a 3-D tensor.
    intensity = h.sum(axis=1)*l.sum(axis=1)/(z.shape[1]*(z.shape[1]-1)/2)
    any_conflict = (h.max(axis=1)>0)&(l.max(axis=1)>0)
    return dispersion, intensity, any_conflict


def score_all(z, domains, reference, ordinal, config, critic):
    uniform = np.full(len(FIELDS),1/len(FIELDS))
    xi = config["weights"]["critic_shrinkage"]
    weights = (1-xi)*uniform+xi*critic
    edges = pair_indices(config["conflict"]["pairs"])
    eta = config["conflict"]["eta"]
    theta = interaction_weights(weights,edges,eta)
    base,q,penalty = quality(z,weights,edges,theta)
    d,k,i = diagnostic_scores(z,weights,config["conflict"]["high"],config["conflict"]["low"])
    mad, r = {}, np.zeros(len(z))
    for g in np.unique(domains):
        selected = reference & (domains == g)
        if not selected.any():
            raise ValueError(f"No A1 reference records in domain {g}")
        median = float(np.median(d[selected]))
        scale = max(float(1.4826*np.median(np.abs(d[selected]-median))), config["conflict"]["mad_floor"])
        mad[str(g)] = {"median":median,"scale":scale}
        r[domains == g]=(d[domains == g]-median)/scale
    scores={"Q_base":base,"Q_rule":q,"penalty":penalty,"dispersion":d,"conflict_intensity":k,
            "has_high_low_difference":i,"within_domain_R":r,"within_domain_outlier":r>3,
            "Q_equal":z @ uniform,"Q_critic":z @ critic,"Q_huber":huber_center(z)}
    # Without independent human or downstream-loss labels, the conservative
    # primary score is the transparent weighted average. Q_rule remains a
    # fully computed conflict-aware candidate and is reported for sensitivity.
    scores["Q_primary"] = base if config.get("primary_score", "Q_base") == "Q_base" else q
    for e in [0,.25,.5,1]:
        scores[f"Q_eta_{e:g}"]=quality(z,weights,edges,interaction_weights(weights,edges,e))[1]
    weak=weights.copy()
    for name in ["modernbert_professionalism","dsir_books","dsir_wiki","dsir_math"]:
        weak[FIELDS.index(name)] *= .25
    weak/=weak.sum()
    scores["Q_relevance_downweighted"]=quality(z,weak,edges,interaction_weights(weak,edges,eta))[1]
    alt=z.copy()
    for j,f in enumerate(MODERN):
        alt[:,FIELDS.index(f)]=np.where(np.isfinite(ordinal[:,j]),ordinal[:,j],alt[:,FIELDS.index(f)])
    scores["Q_argmax"]=quality(alt,weights,edges,theta)[1]
    return scores, {"weights":weights.tolist(),"critic_weights":critic.tolist(),"edges":edges,
                    "theta":theta.tolist(),"eta":eta,"mad":mad,"status":"rule_candidate_not_human_validated"}

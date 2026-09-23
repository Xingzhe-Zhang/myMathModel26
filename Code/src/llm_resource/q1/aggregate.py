"""Domain summaries, fixed-rule bootstrap and extension comparisons."""
import numpy as np
import pandas as pd


def mean_bootstrap(x,repeats,rng):
    """Ordinary record bootstrap conditional on fitted scoring parameters."""
    means=[]
    for _ in range(repeats):
        means.append(float(np.mean(x[rng.integers(0,len(x),len(x))])))
    return np.quantile(means,[.025,.975])


def source_masks(corpus):
    n=len(corpus.records)
    masks={}
    for label in ["A1","A2","A3"]:
        mask=np.zeros(n,bool)
        mask[corpus.members.loc[corpus.members.source==label,"row"].unique()]=True
        masks[label]=mask
    masks["A2_new"]=masks["A2"]&~masks["A1"]
    masks["A3_new"]=masks["A3"]&~masks["A1"]
    masks["union"]=np.ones(n,bool)
    return masks


def summaries(scores,domains,lengths,masks,repeats,rng):
    rows=[]
    scenario_rows=[]
    scenarios=[c for c in scores if c.startswith("Q_")]
    for label,selected in masks.items():
        for g in np.unique(domains[selected]):
            mask=selected&(domains==g);n=int(mask.sum())
            q=scores.loc[mask,"Q_primary"].to_numpy()
            low,high=mean_bootstrap(q,repeats,rng)
            valid_lengths=np.isfinite(lengths[mask])&(lengths[mask]>0)
            wordmean=float(np.average(q[valid_lengths],weights=lengths[mask][valid_lengths])) if valid_lengths.any() else np.nan
            rows.append({"dataset":label,"domain":g,"n_unique":n,"Q_base_mean":scores.loc[mask,"Q_base"].mean(),
                         "Q_rule_mean":scores.loc[mask,"Q_rule"].mean(),"Q_primary_mean":q.mean(),
                         "Q_primary_median":np.median(q),"Q_primary_p10":np.quantile(q,.1),"Q_primary_p90":np.quantile(q,.9),
                         "Q_primary_trimmed_mean":q[(q>=np.quantile(q,.05))&(q<=np.quantile(q,.95))].mean(),
                         "ci_low_primary":low,"ci_high_primary":high,"Q_word_weighted_proxy":wordmean,
                         "word_weight_valid_fraction":valid_lengths.mean(),
                         "high_low_difference_rate":scores.loc[mask,"has_high_low_difference"].mean(),
                         "conflict_intensity_mean":scores.loc[mask,"conflict_intensity"].mean(),
                         "outlier_rate_frozen_A1_MAD":scores.loc[mask,"within_domain_outlier"].mean(),
                         "mean_penalty":scores.loc[mask,"penalty"].mean(),
                         "mean_valid_coverage":scores.loc[mask,"valid_coverage"].mean()})
            for scenario in scenarios:
                scenario_rows.append({"dataset":label,"domain":g,"scenario":scenario,"mean":scores.loc[mask,scenario].mean()})
    summary=pd.DataFrame(rows)
    comparisons=[]
    for g,ext in [("arxiv","A2"),("github","A3")]:
        a=summary[(summary.dataset=="A1")&(summary.domain==g)].iloc[0]
        for target in [ext,ext+"_new","union"]:
            b=summary[(summary.dataset==target)&(summary.domain==g)].iloc[0]
            comparisons.append({"domain":g,"comparison":f"{target}-A1","delta_Q":b.Q_primary_mean-a.Q_primary_mean,
                                "delta_conflict_rate":b.high_low_difference_rate-a.high_low_difference_rate,
                                "delta_conflict_intensity":b.conflict_intensity_mean-a.conflict_intensity_mean,
                                "delta_penalty":b.mean_penalty-a.mean_penalty,
                                "independent_records":target.endswith("_new"),
                                "interpretation":"same-source coverage extension, not independent corpus validation"})
    return summary,pd.DataFrame(scenario_rows),pd.DataFrame(comparisons)

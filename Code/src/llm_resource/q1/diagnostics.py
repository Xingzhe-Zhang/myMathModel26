"""Descriptive conflicts, conditional randomization and exploratory factors.

None of these identify a causal defect or a true training benefit.
"""
import numpy as np
import pandas as pd
from .schema import FIELDS


def correlation(x):
    x = x-x.mean(axis=0)
    norm = np.sqrt((x*x).sum(axis=0))
    return np.divide(x.T@x,norm[:,None]*norm[None,:],out=np.zeros((x.shape[1],x.shape[1])),where=norm[:,None]*norm[None,:]>1e-14)


def spearman_matrix(z):
    return correlation(pd.DataFrame(z).rank().to_numpy())


def adjusted_correlation(z, domains, lengths):
    ranks = pd.DataFrame(z).rank(pct=True).to_numpy()
    loglength = np.log1p(np.maximum(np.nan_to_num(lengths,nan=0),0))
    loglength = (loglength-loglength.mean())/max(loglength.std(),1e-12)
    groups = np.unique(domains)
    design = np.column_stack([np.ones(len(z)),loglength,loglength**2]+[(domains==g).astype(float) for g in groups[1:]])
    residual = ranks-design @ np.linalg.lstsq(design,ranks,rcond=None)[0]
    return correlation(residual)


def length_strata(domains, lengths, boundaries):
    bins = np.digitize(np.log1p(np.maximum(np.nan_to_num(lengths,nan=0),0)),boundaries)
    return np.asarray([f"{g}:{b}" for g,b in zip(domains,bins)])


def pair_matrices(z,strata,high,low):
    h=(z>high).astype(float); l=(z<low).astype(float)
    observed=h.T@l
    expected=np.zeros_like(observed)
    for s in np.unique(strata):
        mask=strata==s
        expected+=np.outer(h[mask].sum(axis=0),l[mask].sum(axis=0))/mask.sum()
    np.fill_diagonal(observed,0); np.fill_diagonal(expected,0)
    return h,l,observed,expected


def pair_table(z,strata,high,low,dataset):
    h,l,observed,expected=pair_matrices(z,strata,high,low)
    unconditional=np.outer(h.mean(axis=0),l.mean(axis=0))*len(z)
    rows=[]
    for a in range(len(FIELDS)):
        for b in range(len(FIELDS)):
            if a==b:continue
            rows.append({"dataset":dataset,"high_field":FIELDS[a],"low_field":FIELDS[b],"n":len(z),
                         "count":int(observed[a,b]),"rate":observed[a,b]/len(z),
                         "expected_count_given_domain_length":expected[a,b],
                         "conditional_lift":observed[a,b]/expected[a,b] if expected[a,b]>0 else np.nan,
                         "unconditional_lift":observed[a,b]/unconditional[a,b] if unconditional[a,b]>0 else np.nan,
                         "excess_rate":(observed[a,b]-expected[a,b])/len(z),
                         "adequate_expected_count":expected[a,b]>=5})
    return pd.DataFrame(rows)


def bh_adjust(p):
    p=np.asarray(p,float); order=np.argsort(p); m=len(p)
    sorted_q=np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1]
    result=np.empty(m);result[order]=np.minimum(sorted_q,1)
    return result


def permutation_pairs(z,domains,strata,config,rng):
    selected=[]
    for g in np.unique(domains):
        indexes=np.flatnonzero(domains==g)
        selected.extend(rng.choice(indexes,min(len(indexes),config["permutation_max_per_domain"]),replace=False))
    selected=np.asarray(selected); z=z[selected]; strata=strata[selected]
    h,l,observed,expected=pair_matrices(z,strata,config["high"],config["low"])
    deviation=np.abs(observed-expected)
    exceed=np.zeros_like(observed)
    masks=[np.flatnonzero(strata==s) for s in np.unique(strata)]
    for _ in range(config["permutation_repeats"]):
        hp=h.copy();lp=l.copy()
        for ids in masks:
            for j in range(len(FIELDS)):
                order=rng.permutation(ids)
                hp[ids,j]=h[order,j];lp[ids,j]=l[order,j]
        null=hp.T@lp
        exceed+=(np.abs(null-expected)>=deviation-1e-10)
    table=pair_table(z,strata,config["high"],config["low"],"A1_reference_permutation_subsample")
    off=~np.eye(len(FIELDS),dtype=bool)
    table["p_two_sided"]=(exceed[off]+1)/(config["permutation_repeats"]+1)
    table["q_bh"]=bh_adjust(table.p_two_sided)
    table["permutations"]=config["permutation_repeats"]
    return table


def varimax(loadings,max_iter=200,tol=1e-8):
    p,k=loadings.shape; rotation=np.eye(k); old=0
    for _ in range(max_iter):
        l=loadings@rotation
        target=loadings.T@(l**3-l@(np.diag((l*l).sum(axis=0))/p))
        u,s,v=np.linalg.svd(target)
        rotation=u@v; objective=s.sum()
        if old>0 and abs(objective-old)<tol*old:break
        old=objective
    return loadings@rotation


def promax(loadings,power=4):
    orthogonal=varimax(loadings)
    if loadings.shape[1]==1:return orthogonal,np.ones((1,1))
    target=np.sign(orthogonal)*np.abs(orthogonal)**power
    rotation=np.linalg.lstsq(orthogonal,target,rcond=None)[0]
    factor_cov=np.linalg.pinv(rotation.T@rotation)
    scales=np.sqrt(np.maximum(np.diag(factor_cov),1e-12))
    pattern=orthogonal@rotation@np.diag(scales)
    phi=factor_cov/scales[:,None]/scales[None,:]
    return pattern,phi


def exploratory_factors(z,domains,config,rng):
    groups=np.unique(domains)
    per=min(config["max_per_domain"],min(int((domains==g).sum()) for g in groups))
    ids=np.concatenate([rng.choice(np.flatnonzero(domains==g),per,replace=False) for g in groups])
    x=z[ids]; active=x.std(axis=0)>1e-10; x=x[:,active]
    if x.shape[1]<2:raise ValueError("Not enough nonconstant indicators for factor analysis")
    r=correlation(x);values=np.linalg.eigvalsh(r)[::-1]
    null=[]
    for _ in range(config["parallel_repeats"]):
        shuffled=np.column_stack([rng.permutation(x[:,j]) for j in range(x.shape[1])])
        null.append(np.linalg.eigvalsh(correlation(shuffled))[::-1])
    cut=np.quantile(null,.95,axis=0)
    count=0
    for observed,threshold in zip(values,cut):
        if observed<=threshold:break
        count+=1
    count=min(count,config["max_factors"],x.shape[1]-1)
    parallel=pd.DataFrame({"component":np.arange(1,len(values)+1),"observed_eigenvalue":values,"permutation_q95":cut})
    if count==0:
        return pd.DataFrame(index=FIELDS),pd.DataFrame(),parallel,{"n":len(x),"factors":0,"note":"No factor exceeds the parallel-analysis reference."}
    # Principal-axis extraction estimates communalities, rather than silently
    # relabeling PCA loadings as a common-factor model.
    h=np.clip(1-1/np.maximum(np.diag(np.linalg.pinv(r)),1),.05,.95)
    converged=False
    for iteration in range(300):
        reduced=r.copy();np.fill_diagonal(reduced,h)
        e,v=np.linalg.eigh(reduced); order=np.argsort(e)[::-1][:count]
        load=v[:,order]*np.sqrt(np.maximum(e[order],0))
        updated=np.clip((load*load).sum(axis=1),0,.995)
        if np.max(np.abs(updated-h))<1e-7:
            converged=True;break
        h=updated
    pattern,phi=promax(load,config["promax_power"])
    expanded=np.zeros((len(FIELDS),count));expanded[active]=pattern
    names=[f"factor_{i+1}" for i in range(count)]
    reconstructed=pattern@phi@pattern.T
    off=~np.eye(len(r),dtype=bool)
    metadata={"n":len(x),"equal_records_per_domain":per,"factors":count,"method":"principal_axis_factoring_with_promax",
              "parallel_analysis":"95th percentile of independently permuted marginal distributions",
              "converged":converged,"iterations":iteration+1,
              "off_diagonal_reconstruction_rmse":float(np.sqrt(np.mean((r-reconstructed)[off]**2))),
              "constant_fields":[f for f,keep in zip(FIELDS,active) if not keep],
              "interpretation":"descriptive covariance structure, not causal explanation"}
    return pd.DataFrame(expanded,index=FIELDS,columns=names),pd.DataFrame(phi,index=names,columns=names),parallel,metadata

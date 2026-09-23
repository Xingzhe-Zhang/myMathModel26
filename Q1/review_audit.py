"""Read-only audit of source attachments; writes review evidence next to this file.

Not a completed Q1 implementation. Uses training-only CV for exploratory benchmarks.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import json, lzma, hashlib, collections
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'real_attachments/A_data_value'
OUT = Path(__file__).with_name('review_evidence.json')
result = {'scope': 'Attachment audits and exploratory train-only CV; no human quality labels, no LightGBM fit.'}

def emit():
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

def load_pair(tag):
    m = pd.read_csv(A / f'regmix_tables/{tag.replace("TAG", "mixture")}.csv').set_index('index')
    y = pd.read_csv(A / f'regmix_tables/{tag.replace("TAG", "pile_loss")}.csv').set_index('index')
    assert m.index.is_unique and y.index.is_unique and set(m.index) == set(y.index)
    y = y.loc[m.index]
    p = m.to_numpy(float)
    assert np.all(p >= 0)
    return m, y, p / p.sum(axis=1, keepdims=True)

tags = ['train_TAG_1m', 'test_TAG_1m', 'test_TAG_60m', 'test_TAG_1B', 'est_TAG_10b', 'est_TAG_70b']
sets = {t: load_pair(t) for t in tags}
names = [s.removeprefix('train_the_pile_') for s in sets[tags[0]][0].columns]
p = sets[tags[0]][2]
y = sets[tags[0]][1].to_numpy(float)
pairs = [(i, j) for i in range(17) for j in range(i+1, 17)]
phi = np.column_stack([p] + [p[:, i]*p[:, j] for i, j in pairs])
sv = np.linalg.svd(phi, compute_uv=False)
support = {names[j]: {'positive_rows': int((p[:, j]>0).sum()), 'max_share': float(p[:, j].max())} for j in range(17)}
pair_counts = [(int(((p[:, i]>0)&(p[:, j]>0)).sum()), names[i], names[j]) for i,j in pairs]
dist = np.abs(p[:, None, :] - p[None, :, :]).sum(axis=2)
np.fill_diagonal(dist, np.inf)
result['design'] = {'shape': list(phi.shape), 'rank': int(np.linalg.matrix_rank(phi)), 'condition_number': float(sv[0]/sv[-1]), 'zero_interaction_columns': [f'{names[i]}*{names[j]}' for i,j in pairs if not np.any(p[:,i]*p[:,j])], 'pair_support_min': sorted(pair_counts)[:10], 'domain_support': support, 'nearest_l1_quantiles': np.quantile(dist.min(axis=1),[0,.1,.5,.9,1]).tolist()}
result['cross_tables'] = {}
for tag,(m,yy,pp) in sets.items():
    dd = np.abs(pp[:,None,:]-p[None,:,:]).sum(axis=2)
    result['cross_tables'][tag] = {'rows':len(pp),'unique_proportions':int(len(np.unique(pp,axis=0))), 'matched_training_proportions':int((dd.min(axis=1)<1e-12).sum())}
for tag in ['est_TAG_10b','est_TAG_70b']:
    m,yy,pp=sets[tag]
    dd=np.abs(pp[:,None,:]-p[None,:,:]).sum(axis=2)
    match=dd.argmin(axis=1)
    source=y[match]; target=yy.to_numpy(float)
    ratios=target/source
    pred=source*np.median(ratios,axis=0)
    row_scales=np.median(ratios,axis=1)
    result['cross_tables'][tag].update({'per_domain_median_ratio':np.median(ratios,axis=0).tolist(),'max_abs_residual_after_domain_scaling':float(np.abs(pred-target).max()),'global_ratio_min_max':[float(ratios.min()),float(ratios.max())], 'per_recipe_scaling_max_abs_residual':float(np.abs(source*row_scales[:,None]-target).max()), 'per_recipe_scale_range':[float(row_scales.min()),float(row_scales.max())]})
print('Mixture structure audit complete',flush=True)
emit()

# Connected-component grouped CV for mixtures within L1 distance 0.01.
parents=list(range(len(p)))
def root(i):
    while parents[i]!=i:
        parents[i]=parents[parents[i]]; i=parents[i]
    return i
for i,j in zip(*np.where(np.triu(dist <= .01,1))):
    parents[root(int(i))]=root(int(j))
groups=np.array([root(i) for i in range(len(p))])
unique=np.unique(groups); np.random.default_rng(20260923).shuffle(unique)
foldmap={int(g):i%5 for i,g in enumerate(unique)}
fold=np.array([foldmap[int(g)] for g in groups])
result['benchmark_protocol']={'seed':20260923,'folds':5,'group_l1_threshold':.01,'groups':len(unique),'selection_metric':'13-output mean squared error on train-only CV','target':'13-domain macro-mean original cross entropy; no target-scale calibration','quadratic_regularization':'RMS feature scaling within fold; interaction penalty factor 1,10,100 relative to linear block','kernel':'RBF ridge on original proportions, bandwidth based on training-fold pairwise distances','limitation':'Exploratory single grouped CV split; not nested final evaluation or proof of best model.'}

def features(x,quad):
    return np.column_stack([x]+[x[:,i]*x[:,j] for i,j in pairs]) if quad else x.copy()

def ridge_predictions(x,yy,xt,quad,ratio,lam):
    f=features(x,quad); ft=features(xt,quad)
    scale=np.sqrt(np.mean(f*f,axis=0)); scale[scale<1e-12]=1
    if quad:scale[17:]*=np.sqrt(ratio)
    f=f/scale; ft=ft/scale
    mean=yy.mean(axis=0)
    b=np.linalg.solve(f.T@f+lam*len(x)*np.eye(f.shape[1]),f.T@(yy-mean))
    return mean+ft@b

def kernel_predictions(x,yy,xt,gamma,lam):
    d=((x[:,None,:]-x[None,:,:])**2).sum(axis=2)
    bandwidth=np.median(d[np.triu_indices(len(x),1)])
    k=np.exp(-gamma*d/max(bandwidth,1e-12))
    kt=np.exp(-gamma*((xt[:,None,:]-x[None,:,:])**2).sum(axis=2)/max(bandwidth,1e-12))
    mean=yy.mean(axis=0)
    return mean+kt@np.linalg.solve(k+lam*len(x)*np.eye(len(x)),yy-mean)

def metrics(actual,pred):
    a=actual.mean(axis=1); b=pred.mean(axis=1)
    rho=lambda u,v:float(np.corrcoef(pd.Series(u).rank(),pd.Series(v).rank())[0,1])
    return {'all_output_rmse':float(np.sqrt(np.mean((actual-pred)**2))), 'macro_loss_spearman':rho(a,b),'mean_domain_spearman':float(np.mean([rho(actual[:,k],pred[:,k]) for k in range(13)])),'macro_loss_regret':float(a[b.argmin()]-a.min()), 'selected_test_row_position':int(b.argmin())}

bench={}
for family in ['linear','quadratic','rbf']:
    best=None
    ratios=[1,10,100] if family=='quadratic' else ([.25,1,4] if family=='rbf' else [1])
    for ratio in ratios:
        for lam in [1e-6,1e-4,.01,.1]:
            err=[]
            for k in range(5):
                tr=fold!=k; va=~tr
                if family=='rbf':pred=kernel_predictions(p[tr],y[tr],p[va],ratio,lam)
                else:pred=ridge_predictions(p[tr],y[tr],p[va],family=='quadratic',ratio,lam)
                err.append(float(np.mean((y[va]-pred)**2)))
            score=float(np.mean(err))
            if best is None or score<best[0]:best=(score,ratio,lam)
    score,ratio,lam=best
    item={'cv_rmse':float(np.sqrt(score)),'ratio_or_gamma':ratio,'lambda':lam,'test':{}}
    for tag in ['test_TAG_1m','test_TAG_60m','test_TAG_1B']:
        _,yy,pp=sets[tag]
        pred=kernel_predictions(p,y,pp,ratio,lam) if family=='rbf' else ridge_predictions(p,y,pp,family=='quadratic',ratio,lam)
        item['test'][tag]=metrics(yy.to_numpy(float),pred)
    bench[family]=item
    print('Benchmark',family,json.dumps(item),flush=True)
result['exploratory_benchmark']=bench
emit()

# Full stream audit: identity overlap and raw-field semantics, not quality fitting.
sample=A/'slimpajama_quality_signal_sample.jsonl.xz'
aux={'id','content','sub_path','_source_domain','_source_path'}
sample_map={}; rows=[]; quality={}; domains=[]; lengths=[]
for path in [sample]+sorted((A/'slimpajama_quality_extended').glob('*.xz')):
    counts=collections.Counter(); shapes=collections.defaultdict(collections.Counter)
    overlap=0; identical=0; total=0; missing=collections.Counter(); ids=set()
    with lzma.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            o=json.loads(line); total+=1
            g=o.get('_source_domain',path.name.split('_')[0]); counts[g]+=1
            signals={k:v for k,v in o.items() if k not in aux}
            key=(g,o['id']); ids.add(key)
            digest=hashlib.sha256(json.dumps(signals,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            for k,v in signals.items():
                shapes[k][len(v) if isinstance(v,list) else 'scalar']+=1
                if v is None:missing[k]+=1
            if path==sample:
                sample_map[key]=digest
                if len(rows)==0:fields=sorted(signals)
                vals=[]
                for k in fields:
                    v=signals[k]
                    vals.extend(v if isinstance(v,list) else [v])
                rows.append(vals); domains.append(g); lengths.append(float(o['rps_doc_word_count']))
            elif key in sample_map:
                overlap+=1; identical+=int(sample_map[key]==digest)
    quality[path.name]={'records':total,'domains':dict(counts),'unique_domain_ids':len(ids),'overlap_with_A1':overlap,'overlap_with_identical_22_fields':identical,'missing':dict(missing),'field_shapes':{k:{str(a):b for a,b in v.items()} for k,v in shapes.items()}}
    print('Quality audit',path.name,total,'overlap',overlap,'identical',identical,flush=True)
result['quality_audit']=quality
counts=collections.Counter(domains)
result['A1_domain_balanced_reference_effective_n']=float(49/sum(1/n for n in counts.values()))
result['derived_checks']={'uniform_independent_22_score_high_low_conflict_probability':1-2*.75**22+.5**22, 'one_zero_21_ones_equal_weight_min_quality':1-2/22, 'unique_domain_ids_across_quality_files':sum(v['unique_domain_ids'] for v in quality.values())-sum(v['overlap_with_A1'] for v in quality.values()),'arxiv_new_records':17523-1419,'github_new_records':203752-10000}
result['notes']=['Q is not fitted because interval utilities, human labels and pair selection are not specified in the proposal.','Full score-field fingerprints verify duplicate observations, not cryptographic identity of absent text.','No external source data or test labels used to fit quality utilities.']
emit()
print('Saved',OUT,flush=True)

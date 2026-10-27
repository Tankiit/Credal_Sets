import numpy as np
GROUPS = [(0,3),(3,5),(5,8),(8,10)]

def decode(scores):
    return np.stack([scores[:,a:b].argmax(1) for a,b in GROUPS],1)

def evaluate(before, after, reference, truth, objects):
    b,p,r,t = map(decode, (before,after,reference,truth))
    rows = np.arange(len(p))
    target = 2*np.asarray(objects)
    unaffected = np.ones(p.shape, bool)
    unaffected[rows,target] = False
    initially_correct = (b==t)&unaffected
    result = {
        'target_success': (p[rows,target]==t[rows,target]),
        'unaffected_accuracy': ((p==t)&unaffected).sum(1)/3,
        'joint_success': (p==t).all(1),
        'edited_image_agreement': (p==r).mean(1),
        'initial_unaffected_correct': initially_correct.sum(1),
        'preserved_correct': ((p==t)&initially_correct).sum(1),
        'source_prediction': b, 'edited_prediction': p,
        'reference_prediction': r, 'target_truth': t,
    }
    return result

def summarize(records, seed=0, repeats=500):
    # Resample independent source scenes, not correlated edits.
    rng = np.random.default_rng(seed)
    ids = sorted({r['scene_id'] for r in records})
    groups = [[r for r in records if r['scene_id']==i] for i in ids]
    keys = ['target_success','unaffected_accuracy','joint_success','edited_image_agreement']
    values = np.array([[np.mean([r[k] for r in g]) for k in keys] for g in groups])
    boot = values[rng.integers(len(ids),size=(repeats,len(ids)))].mean(1)
    out = {k: {'mean':float(values[:,j].mean()), 'scene_bootstrap_ci95':np.quantile(boot[:,j],[.025,.975]).tolist()} for j,k in enumerate(keys)}
    denom = sum(r['initial_unaffected_correct'] for r in records)
    out['preservation_given_initially_correct'] = sum(r['preserved_correct'] for r in records)/denom if denom else None
    out['n_scenes'], out['n_pairs'] = len(ids), len(records)
    return out

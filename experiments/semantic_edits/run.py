import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from .data import scenes, pairs, concepts, features, action, heldout, manifest, render
from .models import Representation, Ridge, LinearEdit, MLPEdit, coordinate
from .metrics import evaluate, summarize

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('results/semantic_edits/pilot'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data-seed', type=int, default=17)
    p.add_argument('--train-scenes', type=int, default=400)
    p.add_argument('--val-scenes', type=int, default=80)
    p.add_argument('--test-scenes', type=int, default=160)
    p.add_argument('--steps', type=int, default=400)
    args = p.parse_args()
    if min(args.train_scenes,args.val_scenes,args.test_scenes,args.steps)<1:
        p.error('scene counts and steps must be positive')
    if args.out.exists():
        p.error('output exists; choose a fresh directory to preserve provenance')
    args.out.mkdir(parents=True)
    base = {s:scenes(n,s,args.data_seed+i) for i,(s,n) in enumerate([
        ('train',args.train_scenes),('val',args.val_scenes),('test',args.test_scenes)])}
    # Reject duplicate rendered scene configurations across base splits.
    # All variants of one source stay together; exact endpoint image overlap is
    # also checked below and validation/test pairs removed if seen earlier.
    batches, seen = {}, set()
    for split in ('train','val','test'):
        raw = pairs(base[split])
        retained = []
        for s in base[split]:
            family = [v for v in raw if v[0].scene_id == s.scene_id]
            hashes = {hashlib.sha256(render(t).tobytes()).hexdigest() for v in family for t in v[:2]}
            if not hashes.intersection(seen):
                retained.extend(family)
        if not retained:
            raise ValueError(f'No pairs remain in {split}; increase scene count')
        seen.update(hashlib.sha256(render(t).tobytes()).hexdigest() for v in retained for t in v[:2])
        batches[split] = retained
    (args.out/'scenes.json').write_text(json.dumps({k:manifest(v) for k,v in base.items()},indent=2))
    config = vars(args).copy(); config['out'] = str(args.out)
    config.update(python=platform.python_version(),numpy=np.__version__,backend='numpy_spatial_pixels',
                  source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    arrays = {}
    for split, pp in batches.items():
        ss,tt,oo,cc = zip(*pp)
        arrays[split] = (features(ss),features(tt),np.stack([concepts(s) for s in ss]),
                         np.stack([concepts(t) for t in tt]),np.stack([action(o,c) for o,c in zip(oo,cc)]))
    summary, all_records = {}, []
    for arch,dim in [('cbm',0),('residual_cbm',16)]:
        # Fit perception on unique TRAIN source and endpoint images only.
        unique = {}
        for s,t,_,_ in batches['train']:
            for v in (s,t): unique[render(v).tobytes()] = v
        fit_scenes = list(unique.values())
        fx,fy = features(fit_scenes),np.stack([concepts(s) for s in fit_scenes])
        rep = Representation(dim,args.seed).fit(fx,fy)
        head = Ridge(.01).fit(rep(fx),fy) # fixed attribute-question reader
        z = {s:(rep(v[0]),rep(v[1])) for s,v in arrays.items()}
        tr,va,te = arrays['train'],arrays['val'],arrays['test']
        linear = LinearEdit().fit(*[z['train'][0],tr[4],z['train'][1],z['val'][0],va[4],z['val'][1]])
        mlp = MLPEdit(args.seed,args.steps).fit(z['train'][0],tr[4],z['train'][1],z['val'][0],va[4],z['val'][1])
        original,ref = z['test']
        outputs = {'identity':original,'coordinate':coordinate(original,te[4]),
                   'linear':linear(original,te[4]),'mlp':mlp(original,te[4]),'edited_image_reference':ref}
        np.savez_compressed(args.out/f'{arch}_checkpoint.npz',
            concept_mean=rep.concepts.mean,concept_target_mean=rep.concepts.target_mean,concept_weight=rep.concepts.weight,
            pixel_mean=rep.mean,projection=rep.projection,head_mean=head.mean,head_target_mean=head.target_mean,head_weight=head.weight,
            linear_mean=linear.model.mean,linear_target_mean=linear.model.target_mean,linear_weight=linear.model.weight,
            mlp_mean=mlp.mean,mlp_scale=mlp.scale,**{f'mlp_p{i}':v for i,v in enumerate(mlp.params)})
        summary[arch] = {'mlp_selected_step':mlp.best_step,'methods':{}}
        np.savez_compressed(args.out/f'{arch}_test_arrays.npz',original=original,reference=ref,actions=te[4],truth=te[3],**{f'edited_{k}':v for k,v in outputs.items()})
        for method,edited in outputs.items():
            metrics = evaluate(head(original),head(edited),head(ref),te[3],[v[2] for v in batches['test']])
            records = []
            for i,(s,t,o,c) in enumerate(batches['test']):
                r = dict(scene_id=s.scene_id,pair_id=f'{s.scene_id}-o{o}-c{c}',architecture=arch,method=method,
                         seed=args.seed,object=o,target_color=c,heldout_combination=heldout(s) or heldout(t))
                r.update({k:np.asarray(v[i]).tolist() for k,v in metrics.items()})
                records.append(r)
            all_records.extend(records)
            strata = {'all':records,'seen_combinations':[r for r in records if not r['heldout_combination']],
                      'heldout_combinations':[r for r in records if r['heldout_combination']]}
            summary[arch]['methods'][method] = {k:summarize(v,args.seed) for k,v in strata.items() if v}
        print(arch, {m:round(v['all']['joint_success']['mean'],3) for m,v in summary[arch]['methods'].items()},flush=True)
    (args.out/'metrics.json').write_text(json.dumps(summary,indent=2))
    with (args.out/'predictions.jsonl').open('w') as f:
        for r in all_records: f.write(json.dumps(r)+'\n')
    with (args.out/'pairs.jsonl').open('w') as f:
        for split, pp in batches.items():
            for s,t,o,c in pp: f.write(json.dumps({'split':split,'source':manifest([s])[0], 'target':manifest([t])[0], 'object':o,'color':c})+'\n')
    from PIL import Image
    preview = Image.new('RGB',(128,32*8),'white')
    for i,(s,t,_,_) in enumerate(batches['test'][:8]):
        preview.paste(render(s),(0,i*32)); preview.paste(render(t),(64,i*32))
    preview.save(args.out/'pairs.png')

if __name__ == '__main__': main()

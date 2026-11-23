import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SEEDS=(42,43,44)
KEYS=('psnr','sam','ergas')

def main():
    data={}
    for model in ('baseline','deform_stage3'):
        for seed in SEEDS:
            for sf in (4,8):
                p=ROOT/'Checkpoint/v3_balanced_scales'/f'{model}_seed{seed}'/('metrics.json' if sf==4 else 'metrics_x8.json')
                report=json.loads(p.read_text());assert report['epoch']==600 and len(report['per_scene'])==12
                data[model,seed,sf]={r['scene']:r for r in report['per_scene']}
    scenes=list(data['baseline',42,4]);rows=[];summary={}
    for sf in (4,8):
        counts={k:0 for k in KEYS};all_gain=0;seed_consistent={k:0 for k in KEYS}
        for scene in scenes:
            row=dict(scene=scene,sf=sf)
            for k in KEYS:
                base=[data['baseline',s,sf][scene][k] for s in SEEDS]
                stage=[data['deform_stage3',s,sf][scene][k] for s in SEEDS]
                delta=[b-a for a,b in zip(base,stage)]
                row['baseline_'+k]=statistics.mean(base);row['stage3_'+k]=statistics.mean(stage)
                row['delta_'+k]=statistics.mean(delta);row['delta_'+k+'_std']=statistics.stdev(delta)
                sign=1 if k=='psnr' else -1
                counts[k]+=sign*row['delta_'+k]>0
                seed_consistent[k]+=all(sign*d>0 for d in delta)
            all_gain+=row['delta_psnr']>0 and row['delta_sam']<0 and row['delta_ergas']<0
            rows.append(row)
        summary[sf]=dict(improved_scene_counts=counts,all_three_metrics_improved=all_gain,improved_in_each_of_three_seeds=seed_consistent,
            mean_delta={k:statistics.mean(r['delta_'+k] for r in rows if r['sf']==sf) for k in KEYS})
    with (OUT/'per_scene_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'scene_summary.json').write_text(json.dumps(dict(seeds=SEEDS,delta_definition='Stage3 minus baseline; means over 3 paired training seeds',results=summary),indent=2)+'\n')
    lines=['| 场景 | ×4 ΔPSNR↑ | ΔSAM↓ | ΔERGAS↓ | ×8 ΔPSNR↑ | ΔSAM↓ | ΔERGAS↓ |','|---|---:|---:|---:|---:|---:|---:|']
    for scene in scenes:
        vals=[]
        for sf in (4,8):
            r=next(r for r in rows if r['scene']==scene and r['sf']==sf)
            vals += [f"{r['delta_'+k]:+.4f}" for k in KEYS]
        lines.append('| '+' | '.join([scene,*vals])+' |')
    (OUT/'per_scene_comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2));print('\n'.join(lines))
if __name__=='__main__':main()

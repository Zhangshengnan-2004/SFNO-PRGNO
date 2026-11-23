"""Freeze V3 source snapshots against the active, unmodified SFNO data protocol."""
import copy,json
from protocol import ROOT,PROTOCOL,DEFAULTS,sha256,source_hashes,atomic_json


def main():
    if (ROOT/'protocol.json').exists():raise FileExistsError('Frozen V3 protocol already exists')
    lineage=json.loads((ROOT/'lineage.json').read_text());base=ROOT.parent/'sfno_restart_20261111'
    if sha256(base/'protocol.json')!=lineage['baseline_protocol_sha256']:raise ValueError('Baseline protocol changed')
    original=json.loads((base/'protocol.json').read_text())
    if original['defaults']!=DEFAULTS:raise ValueError('Unmatched optimizer or training budget')
    for name,digest in original['source_hashes'].items():
        if sha256(base/name)!=digest:raise ValueError(f'Frozen baseline source changed:{name}')
    for spec in original['datasets'].values():
        if sha256(ROOT/spec['response_file'])!=spec['response_sha256']:raise ValueError('Unmatched SRF')
    for source,digest in lineage['source_snapshot'].items():
        if sha256(ROOT/'model'/source.split('/')[-1])!=digest:raise ValueError('V3 snapshot differs from recorded source')
    p=copy.deepcopy(original);p.pop('paper_alignment_sha256',None)
    p.update(protocol=PROTOCOL,model='V3-stage3',ablation='deform_stage3',guidance_mode='physics',defaults=DEFAULTS,
             lineage_sha256=sha256(ROOT/'lineage.json'),source_hashes=source_hashes(),
             comparison=dict(baseline_experiment=str(base),baseline_protocol_sha256=lineage['baseline_protocol_sha256']))
    p['implementation'].update(deterministic=False,deterministic_algorithms='warn_only; CUDA grid_sample/antialias backward nondeterministic',
                               sdpa='math',shared_pr_within_effective_batch=False,
                               initialization='native V3 constructors; retain zero physics heads and deformable offset initialization',
                               spatial_operator='matched periodic sigma2 sf-sized FFT forward and exact adjoint')
    p['assumptions']=[s for s in p['assumptions'] if not ('SFNO only' in s or 'SFNO constructor' in s or 'identical PR' in s)]
    p['assumptions']+=['V3-stage3 ablation=deform_stage3, physics guidance, both observation residuals enabled; native V3 weights from scratch.',
                        'Same corrected data, loss, budget, optimizer and final evaluation as the active SFNO baseline.',
                        'CUDA grid_sample/antialias backward permits numerical nondeterminism. Seeded resume is checked within tolerance and its measured differences are recorded, not claimed bitwise exact.']
    atomic_json(ROOT/'protocol.json',p);print('Frozen V3-stage3 protocol prepared against',base)


if __name__=='__main__':main()

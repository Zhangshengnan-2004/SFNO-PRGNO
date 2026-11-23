"""Frozen approximation: paper-known settings plus explicit unresolved assumptions."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
PROTOCOL='v3_stage3_matched_cachefix_v2_20261112'
DEFAULTS=dict(epochs=600,samples=1000,micro_batch=5,effective_batch=50,lr=2e-4,lr_gamma=.95,
              lr_decay_epochs=6,quality_min_optimizer_steps=2000,quality_patience=3,
              train_sf=4,test_seed=1,patch_size=64,save_every=120)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def source_hashes():
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(ROOT.rglob('*.py'))}


def load_protocol():
    p=ROOT/'protocol.json'
    data=json.loads(p.read_text())
    if data['protocol']!=PROTOCOL:raise ValueError('Protocol mismatch')
    if source_hashes()!=data['source_hashes']:raise ValueError('Frozen source changed; create a new experiment')
    if sha256(ROOT/'lineage.json')!=data['lineage_sha256']:raise ValueError('Frozen comparison lineage changed')
    for ds in data['datasets'].values():
        if sha256(ROOT/ds['response_file'])!=ds['response_sha256']:raise ValueError('Frozen SRF changed')
    return data


def output_dir(dataset,seed):
    return ROOT/'Checkpoint'/f'{dataset}_v3_stage3_x4_seed{seed}'

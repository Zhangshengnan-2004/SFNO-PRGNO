"""Freeze response sources, known settings and explicit assumptions before training."""
import json
import re
import numpy as np
import scipy.io as sio
from protocol import ROOT, PROJECT, PROTOCOL, DEFAULTS, sha256, atomic_json, source_hashes
from data import load_cube,PAVIA_TEST,CHIK_TRAIN,CHIK_TEST


def main():
    if (ROOT/'protocol.json').exists():raise FileExistsError('Frozen protocol exists')
    refpath=PROJECT.parent/'SHOTUN-main/data/pavia/paviac_256_s8.mat'
    sample=sio.loadmat(refpath)
    pavia=load_cube('pavia')
    srf=sample['srf']
    if srf.shape!=(102,4) or (srf<0).any() or not np.allclose(srf.sum(0),1):raise ValueError('Unexpected four-band SRF')
    reconstruction=float(np.abs(sample['REF']@srf-sample['MSI']).max())
    alignment=float(np.abs(pavia[246:502,235:491].astype('float64')/8000-sample['REF']).max())
    if reconstruction>1e-12 or alignment>1e-12:raise ValueError('Pavia band/crop/normalization correspondence failed')
    np.save(ROOT/'response_pavia.npy',srf.T.astype('float32'))
    canon_path=ROOT/'references/canon_5d_mark_ii.json'
    hdr=PROJECT/'Chikusei/HyperspecVNIR_Chikusei_20140729.hdr'
    waves=np.array([float(v) for v in re.search(r'wavelength\s*=\s*\{([^}]+)\}',hdr.read_text(),re.S).group(1).split(',')])*1000
    values=json.loads(canon_path.read_text())['spectral_data']['data']['main']
    measured=np.array(sorted(float(k) for k in values))
    response=np.array([values[str(int(w))] for w in measured])
    r=np.stack([np.interp(waves,measured,response[:,c],left=0,right=0) for c in range(3)])
    r=(r/r.sum(1,keepdims=True)).astype('float32');np.save(ROOT/'response_chikusei.npy',r)
    raw_pavia=PROJECT/'Pavia/Data/Pavia.mat';raw_chik=PROJECT/'Chikusei/HyperspecVNIR_Chikusei_20140729.bsq'
    datasets={
        'pavia':dict(hsi_channels=102,msi_channels=4,normalization=8000,response_file='response_pavia.npy',
            response_sha256=sha256(ROOT/'response_pavia.npy'),raw_sha256={str(raw_pavia):sha256(raw_pavia)},
            train_rectangles_yxhw=[[0,0,704,715]],test_origins_yx=PAVIA_TEST,test_size={'4':192,'8':192},
            response_source=str(refpath),response_source_sha256=sha256(refpath),
            verification=dict(ref_crop_origin=[246,235],ref_crop_max_abs=alignment,msi_reconstruction_max_abs=reconstruction,
                note='Only validates already-training-region crop; this 256px sample is not used as a test scene'),
            physical_sensor='unconfirmed; reusable supplied 102x4 matrix, not claimed as SFNO author SRF'),
        'chikusei':dict(hsi_channels=128,msi_channels=3,normalization=10000,response_file='response_chikusei.npy',
            response_sha256=sha256(ROOT/'response_chikusei.npy'),raw_sha256={str(raw_chik):sha256(raw_chik),str(hdr):sha256(hdr)},
            train_rectangles_yxhw=[[y,x,500,500] for y,x in CHIK_TRAIN],test_origins_yx=CHIK_TEST,
            test_size={'4':500,'8':496},response_source='AcademySoftwareFoundation/rawtoaces-data Canon EOS 5D Mark II',
            response_source_sha256=sha256(canon_path),response_interpolation='HDR wavelengths; linear; zero outside380-780nm; row sum1')}
    for spec in datasets.values():
        for y,x in spec['test_origins_yx']:
            size=spec['test_size']['4']
            for ty,tx,th,tw in spec['train_rectangles_yxhw']:
                if not(y+size<=ty or ty+th<=y or x+size<=tx or tx+tw<=x):raise ValueError('Spatial overlap')
    assumptions=[
        'Approximate reproduction; SFNO-author MSI, exact test coordinates and full training configuration unavailable.',
        '600 epochs x1000 patches, effective batch50:12000 Adam updates per seed. Budget explicitly requested by user, not confirmed author budget. All six jobs restart from scratch.',
        'Paper-known optimizer: Adam beta0.9/0.999 eps1e-8, initial2e-4, decayfactor0.95. Decay interval of six epochs remains an explicit implementation choice; the wording does not supply a complete executable schedule.',
        'Effective batch50 uses microbatch5 x10 with identical PR permutations within each effective batch. Small true-batch SFNO gradient equivalence is required; GPU arithmetic is not claimed bit-identical to a true batch50.',
        'Native SFNO constructor defaults retained; no blanket He reset. The previous all-layer He reset caused severe initial activation amplification; author layer-specific initialization remains unconfirmed.',
        'Technical preflight only; no temporary convergence pilots. Formal jobs start from scratch and persist every epoch. Numerical errors stop immediately; initial probe L1<=2, epoch1 L1<=1, L1<=0.1 after500 Adam updates.',
        'Convergence monitoring uses training-region probes every20epochs. After2000 Adam updates,3 consecutive checks failing to improve both L1 and PSNR over Bicubic stop the queue after saving the checkpoint. No10-percent improvement gate or40-epoch cutoff. No final-test-based tuning.',
        'Per-image GT-nonzero masked L1 chosen as baseline objective; paper training loss not verified; no joint spectral/relative terms.',
        'Fixed normalization8000/10000 and current disjoint spatial splits retained; not tuned against paper/test results.',
        'Chikusei eight test blocks follow4.1;4.2 says six; exact author test set remains unresolved.',
        'Periodic sigma2 degradation, kernel size sf, half-pixel center, phase sf/2-1; boundary details not author-confirmed.',
        'No added noise; test invalid pixels nearest-filled for input, excluded from common metrics.',
        'Final checkpoint selected by fixed budget only; no test-based early stopping/model selection.',
        'SFNO only; V3-stage3 deferred until this protocol is reviewed.']
    atomic_json(ROOT/'protocol.json',dict(protocol=PROTOCOL,model='SFNO',defaults=DEFAULTS,train_seeds=[42,43,44],
        known_paper_settings=dict(patch=64,effective_batch=50,optimizer='Adam',betas=[.9,.999],eps=1e-8,initial_lr=2e-4,gamma=.95,init='He'),
        implementation=dict(precision='FP32',amp=False,workers=0,deterministic=True,loss='mean per-image GT-nonzero masked L1',
            initialization='native SFNO/PyTorch constructor defaults; no blanket reinitialization',
            schedule='lr(epoch_index)=2e-4*0.95**floor(epoch_index/6); six-epoch interval is an unconfirmed implementation choice',
            lr_decay_epochs=DEFAULTS['lr_decay_epochs'],shared_pr_within_effective_batch=True,
            optimizer_updates_per_epoch=DEFAULTS['samples']//DEFAULTS['effective_batch'],
            total_optimizer_updates=DEFAULTS['epochs']*DEFAULTS['samples']//DEFAULTS['effective_batch'],
            total_patches_per_seed=DEFAULTS['epochs']*DEFAULTS['samples'],
            eval='common_hsi_eval_v1; test_seed1 independent of train seed; x4 trained, x4/x8 evaluated'),
        datasets=datasets,assumptions=assumptions,paper_alignment_sha256=sha256(ROOT/'paper_alignment.json'),
        references=dict(dtdnml_simulation_url='https://github.com/Shawn-H-Wang/DTDNML/blob/1a19a8722924e6ed7f310bff3bbb80eef2036df3/simulation/simulation.m',
            dtdnml_simulation_sha256=sha256(ROOT/'references/dtdnml_simulation.m'),
            note='Upstream simulation loads external SRF without naming sensor; only supplied matrix is reused, not its sigma0.5/noise pipeline'),
        source_hashes=source_hashes()))
    print('Frozen protocol prepared; Pavia REF/SRF match max errors:',alignment,reconstruction)


if __name__=='__main__':main()

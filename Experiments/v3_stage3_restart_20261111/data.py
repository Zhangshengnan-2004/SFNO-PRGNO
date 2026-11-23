"""Native cubes, fixed disjoint splits, known-response synthetic MSI, sigma2 blur."""
import random
import re
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from protocol import ROOT, PROJECT, sha256


PAVIA_TEST=[(712,0),(712,192),(904,0),(904,192)]
CHIK_TRAIN=[(y,x) for y in (0,500,1000) for x in (0,500,1000,1500)]
CHIK_TEST=[(y,x) for y in (1500,2000) for x in (0,500,1000,1500)]


def load_cube(dataset):
    if dataset=='pavia':
        path=PROJECT/'Pavia/Data/Pavia.mat'
        raw=sio.loadmat(path)['pavia']
        if raw.shape!=(1096,715,102) or not np.isfinite(raw).all() or raw.min()<0 or raw.max()>8000:
            raise ValueError('Unexpected Pavia source')
        return raw
    path=PROJECT/'Chikusei/HyperspecVNIR_Chikusei_20140729.bsq'
    header=path.with_suffix('.hdr').read_text()
    for key,value in {'samples':'2335','lines':'2517','bands':'128','data type':'2','interleave':'BSQ',
                      'byte order':'0','header offset':'0','reflectance scale factor':'10000.0','data ignore value':'15000.0'}.items():
        match=re.search(r'^'+re.escape(key)+r'\s*=\s*(\S+)',header,re.M|re.I)
        if not match or match.group(1).lower()!=value.lower():raise ValueError(f'Unsupported ENVI {key}')
    if path.stat().st_size!=128*2517*2335*2:raise ValueError('Unexpected BSQ length')
    return np.memmap(path,dtype='<i2',mode='r',shape=(128,2517,2335)).transpose(1,2,0)


def validate_sources(spec):
    for path,digest in spec['raw_sha256'].items():
        if sha256(path)!=digest:raise ValueError(f'Raw data changed: {path}')


def normalization(dataset):return 8000. if dataset=='pavia' else 10000.


def degrade(x,sf):
    coord=torch.arange(sf,device=x.device,dtype=x.dtype)-(sf-1)/2
    g=torch.exp(-coord.square()/8)
    k=g[:,None]*g[None,:];k=k/k.sum()
    left,right=sf//2-1,sf//2
    padded=F.pad(x,(left,right,left,right),mode='circular')
    blur=F.conv2d(padded,k[None,None].expand(x.shape[1],1,sf,sf),groups=x.shape[1])
    phase=sf//2-1
    return blur[:,:,phase::sf,phase::sf]


def observations(gt,sf,response):
    return degrade(gt,sf),torch.einsum('mc,bchw->bmhw',response,gt)


class TrainingPatches(Dataset):
    def __init__(self,cube,dataset,samples):self.cube,self.dataset,self.samples=cube,dataset,samples
    def __len__(self):return self.samples
    def __getitem__(self,index):
        limit=normalization(self.dataset)
        for _ in range(10000):
            if self.dataset=='pavia':
                y,x=random.randrange(704-64+1),random.randrange(715-64+1)
            else:
                ty,tx=random.choice(CHIK_TRAIN)
                y,x=ty+random.randrange(437),tx+random.randrange(437)
            raw=self.cube[y:y+64,x:x+64]
            if ((raw>=0)&(raw<=limit)).all() and (raw>0).any():break
        else:raise RuntimeError('Unable to sample nonempty valid training patch')
        patch=np.rot90(raw.astype('float32')/limit,random.randrange(4))
        if random.random()<.5:patch=patch[::-1]
        if random.random()<.5:patch=patch[:,::-1]
        return torch.from_numpy(patch.copy()).permute(2,0,1).contiguous()


def test_patches(cube,dataset,sf):
    size=192 if dataset=='pavia' else 500//sf*sf
    tiles=PAVIA_TEST if dataset=='pavia' else CHIK_TEST
    limit=normalization(dataset)
    for index,(y,x) in enumerate(tiles,1):
        patch=cube[y:y+size,x:x+size].astype('float32')
        bad=(patch<0)|(patch>limit);valid=~bad.any(2)
        if bad.any():
            from scipy.ndimage import distance_transform_edt
            for c in np.flatnonzero(bad.any(axis=(0,1))):
                nearest=distance_transform_edt(bad[:,:,c],return_distances=False,return_indices=True)
                patch[:,:,c][bad[:,:,c]]=patch[:,:,c][tuple(nearest[:,bad[:,:,c]])]
        patch/=limit
        yield dict(index=index,y=y,x=x,size=size),torch.from_numpy(patch).permute(2,0,1)[None].contiguous(),valid

"""Native geometry-aware input decoding. Only requested labels are read."""
import gzip, io, json, warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings('ignore',category=NotGeoreferencedWarning)
import numpy as np
from scipy.ndimage import zoom
from .source import NativeReader, DATA

def nifti(blob):
    import nibabel as nib
    raw=gzip.decompress(blob) if blob[:2]==b'\x1f\x8b' else blob
    im=nib.Nifti1Image.from_bytes(raw)
    return np.asarray(im.dataobj),np.asarray(im.affine)

def mhd(header,raw):
    h={k.strip():v.strip() for k,v in (s.split('=',1) for s in header.decode().splitlines() if '=' in s)}
    shape=np.array([int(v) for v in h['DimSize'].split()])
    dtype={'MET_SHORT':'<i2','MET_USHORT':'<u2','MET_FLOAT':'<f4'}[h['ElementType']]
    x=np.frombuffer(raw,dtype=dtype).reshape(tuple(shape[::-1])).transpose(2,1,0)
    aff=np.eye(4);aff[:3,:3]=np.array([float(v) for v in h.get('TransformMatrix','1 0 0 0 1 0 0 0 1').split()]).reshape(3,3)@np.diag([float(v) for v in h['ElementSpacing'].split()])
    aff[:3,3]=[float(v) for v in h.get('Offset',h.get('Position',h.get('Origin','0 0 0'))).split()]
    return x,aff

def decode(row,reader,labels=False):
    task=row['task_id'];s=reader.read(row,labels=labels);xs=s['inputs'];ys=s.get('labels',[])
    if task=='MED_02':
        volumes = [nifti(blob) for blob in xs]
        if len(volumes) != 4:
            raise ValueError('BraTS requires T1, T1ce, T2, FLAIR')
        shape, aff = volumes[0][0].shape, volumes[0][1]
        if len(shape) != 3 or any(x.shape != shape or not np.allclose(a, aff, atol=1e-4)
                                  for x, a in volumes):
            raise ValueError('BraTS modalities must share their native grid')
        y = None
        if labels:
            y, ya = nifti(ys[0])
            if y.shape != shape or not np.allclose(ya, aff, atol=1e-4):
                raise ValueError('BraTS mask grid differs from MRI')
            if not np.isin(y, [0, 1, 2, 4]).all():
                raise ValueError('Unexpected native BraTS label')
            y = y.astype(np.int16)
        return [dict(x=np.stack([x.astype(np.float32) for x, _ in volumes]), y=y, affine=aff)]
    if task=='RS_01':
        x=np.asarray(xs[0],dtype=np.float32).transpose(2,0,1)
        return [dict(x=x,y=np.asarray(ys[0],dtype=np.int16) if labels else None,affine=None)]
    if task=='RS_02':
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.warp import reproject, Resampling
        # Reference grid comes from the 492 nm input, never from a hidden mask.
        with MemoryFile(xs[1]) as mem:
            with mem.open() as ds: shape=(ds.height,ds.width);transform=ds.transform;crs=ds.crs
        bands=[]
        for blob in xs:
            with MemoryFile(blob) as mem:
                with mem.open() as ds:
                    a=np.empty(shape,np.float32)
                    if ds.transform.is_identity or transform.is_identity or ds.crs is None or crs is None:
                        # Published MADOS patches often have no geotransform.
                        # Follow the author's same-footprint nearest-neighbour stacking.
                        a=ds.read(1,out_shape=shape,resampling=Resampling.nearest).astype(np.float32)
                    else:
                        reproject(ds.read(1),a,src_transform=ds.transform,src_crs=ds.crs,dst_transform=transform,dst_crs=crs,resampling=Resampling.nearest)
                    bands.append(a)
        y=None
        if labels:
            with MemoryFile(ys[0]) as mem:
                with mem.open() as ds:
                    assert ds.shape==shape and ds.transform.almost_equals(transform)
                    y=ds.read(1).astype(np.int16)-1
        return [dict(x=np.stack(bands),y=y,affine=None)]
    if task=='MED_08':
        from PIL import Image
        x=np.asarray(Image.open(io.BytesIO(xs[0])).convert('L'),dtype=np.float32)[None]/255
        y=(np.asarray(Image.open(io.BytesIO(ys[0])))>0).astype(np.int16) if labels else None
        return [dict(x=x,y=y,affine=None)]
    if task=='MED_05':
        return [dict(x=np.asarray(xs[0],np.float32).transpose(2,0,1)/255,y=np.asarray(ys[0],np.int32) if labels else None,affine=None)]
    if task=='MED_04':
        x,aff=mhd(*xs)
        return [dict(x=x,affine=aff,y=ys if labels else None)]
    result=[]
    for i,blob in enumerate(xs):
        x,aff=nifti(blob);y=None
        if labels:
            y,ya=nifti(ys[i]);assert x.shape==y.shape and np.allclose(aff,ya,atol=1e-4)
            y=y.astype(np.int16)
        result.append(dict(x=x,affine=aff,y=y))
    return result

def processed(task,part,spacing):
    x=part['x'];y=part['y'];aff=part['affine']
    if aff is None:
        x=np.nan_to_num(x.astype(np.float32),nan=0,posinf=0,neginf=0)
        if task in ('RS_01','RS_02'):
            # Per-input robust scaling: neither Q nor held-out labels enter statistics.
            mean=x.mean((1,2),keepdims=True);std=x.std((1,2),keepdims=True)
            x=np.clip((x-mean)/np.maximum(std,1e-6),-10,10)
        return x.astype(np.float16),y
    old=np.array(x.shape);voxel=np.linalg.norm(aff[:3,:3],axis=0)
    shape=np.maximum(4,np.rint((old-1)*voxel/spacing).astype(int)+1)
    x=zoom(x.astype(np.float32),shape/old,order=1,prefilter=False)
    if task in ('MED_03','MED_04','MED_07'):
        low,high=(-1000,400) if task=='MED_04' else (-200,1000)
        x=(np.clip(x,low,high)-low)/(high-low)
    else:
        finite=x[np.isfinite(x)];lo,hi=np.percentile(finite,[1,99])
        x=(np.clip(x,lo,hi)-lo)/max(hi-lo,1e-6)
    x=np.nan_to_num(x).astype(np.float16)[None]
    if y is not None and task!='MED_04':y=zoom(y,shape/old,order=0,prefilter=False).astype(np.int16)
    if y is not None and task=='MED_04':
        annotations=y;target=np.zeros(shape,np.int16)
        for j in [1,0]:
            for n in annotations[j]:
                world=np.array([float(n[k]) for k in ['coordX','coordY','coordZ']])
                center=np.linalg.solve(aff[:3,:3],world-aff[:3,3])*(shape-1)/(old-1)
                radius=max(float(n['diameter_mm'])/2,1)/spacing
                lo=np.maximum(0,np.floor(center-radius-1).astype(int));hi=np.minimum(shape,np.ceil(center+radius+2).astype(int))
                if np.any(lo>=hi):continue
                sl=tuple(slice(a,b) for a,b in zip(lo,hi));grid=np.ogrid[sl]
                mask=sum((g-c)**2 for g,c in zip(grid,center))<=radius**2
                target[sl][mask]=1 if j==0 else -1
        y=target
    return x,y

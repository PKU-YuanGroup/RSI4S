"""Decode native attachments through an explicit input-only schema."""
import hashlib
import pickle
from pathlib import Path


def public_bytes(source):
    source=Path(source)
    raw=source.read_bytes()
    if source.suffix=='.graphdata':
        import numpy as np
        graph=pickle.loads(raw)
        if not isinstance(graph,list) or len(graph) not in (6,7):raise ValueError('Unknown MOF graph schema')
        if not isinstance(graph[0],str) or graph[0]!=source.stem:raise ValueError('MOF structure ID mismatch')
        if any(not isinstance(graph[i],np.ndarray) for i in (1,2,3,5)) or not isinstance(graph[4],list):
            raise ValueError('Unknown MOF graph input fields')
        if any(graph[i].dtype.hasobject for i in (1,2,3,5)):raise ValueError('Unexpected object-valued graph field')
        # The seventh field contains band-gap target values in native QMOF
        # files, even for test. hMOF uses a metadata string in this position.
        # Keep tuple-unpacking compatibility, with an explicit missing target.
        return pickle.dumps(graph[:6]+[None],protocol=4)
    if source.suffix=='.griddata16':
        import numpy as np
        grid=pickle.loads(raw)
        if not isinstance(grid,np.ndarray) or grid.dtype.hasobject or grid.ndim!=1:
            raise ValueError('Unknown MOF grid schema')
    elif source.suffix=='.grid':
        lines=raw.decode().splitlines()
        if len(lines)!=3 or [s.split()[0] for s in lines]!=['CELL_PARAMETERS','CELL_ANGLES','GRID_NUMBERS']:
            raise ValueError('Unknown MOF grid metadata schema')
    else:raise ValueError('Unreviewed attachment type')
    return raw


def materialize(source,destination,expected_source_hash,*,map_compresslevel=1,deadline=None):
    from benchmarks.naturebench_active.preparation import check, file_hash
    check(deadline)
    source=Path(source);destination=Path(destination)
    if source.name.endswith('.map.gz'):
        import gzip, shutil, tempfile
        def filehash(path):
            return file_hash(path,deadline)
        if filehash(source)!=expected_source_hash:raise ValueError('Map source changed')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            # Compression is transport only. Resume a previously sanitized
            # map after checking every decompressed byte against its source.
            with gzip.open(source,'rb') as inp,gzip.open(destination,'rb') as existing:
                header=inp.read(1024)
                if len(header)!=1024 or header[208:212]!=b'MAP ':raise ValueError('Invalid MRC input')
                if existing.read(1024)!=header[:220]+bytes(804):raise ValueError('Materialized map header changed')
                while True:
                    check(deadline)
                    block=inp.read(4*1024*1024)
                    if existing.read(len(block) if block else 1)!=block:raise ValueError('Materialized map voxels changed')
                    if not block:break
            return filehash(destination)
        with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as raw:
            temporary=Path(raw.name)
            try:
                # Fast lossless compression keeps research input preparation
                # from spending its stage budget on high-level recompression.
                with gzip.open(source,'rb') as inp,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=map_compresslevel) as out:
                    header=inp.read(1024)
                    if len(header)!=1024 or header[208:212]!=b'MAP ':raise ValueError('Invalid MRC input')
                    # Strip annotation labels and gzip filenames/timestamps;
                    # numeric grid fields and every voxel are unchanged.
                    out.write(header[:220]+bytes(804))
                    while True:
                        check(deadline)
                        block=inp.read(4*1024*1024)
                        if not block:break
                        out.write(block)
                raw.flush();result=filehash(temporary)
                if destination.exists():
                    if filehash(destination)!=result:raise ValueError('Materialized map changed')
                else:
                    temporary.replace(destination);destination.chmod(0o444)
            finally:
                if temporary.exists():temporary.unlink()
        return result
    if file_hash(source,deadline)!=expected_source_hash:
        raise ValueError('Native input integrity check failed')
    check(deadline)
    payload=public_bytes(source)
    check(deadline)
    if destination.exists():
        if destination.read_bytes()!=payload:raise ValueError('Materialized input changed')
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(payload);destination.chmod(0o444)
    check(deadline)
    return hashlib.sha256(payload).hexdigest()

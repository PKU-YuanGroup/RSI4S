"""Contracts for the eight tasks reviewed in nb_pending_tasks_Lh.md.

These tasks preserve native row membership, including documented overlaps. A
row group is an acquisition unit, not a claim of biological independence.
"""
from pathlib import Path
import json
import shutil

from benchmarks.naturebench_active.common import digest, sha256

YEAST = 'LIFE_05'
POCKET = 'CHEM_06'
RNA = 'LIFE_11'
OPTIMIZATION = 'CHEM_08'
ECD = 'CHEM_11'
COVID = 'MED_09'
RNA_GENERATION = 's41592-023-02148-8'
MOLECULE_GENERATION = 'CHEM_04'
TASKS = (YEAST, POCKET, RNA, OPTIMIZATION, ECD, COVID, RNA_GENERATION, MOLECULE_GENERATION)
FEATURE_TASKS = (POCKET, ECD)
LOSS_TASKS = (POCKET, RNA_GENERATION, MOLECULE_GENERATION)
FAMILIES_18 = (
    'RF00001', 'RF00003', 'RF00004', 'RF00005', 'RF00017', 'RF00029',
    'RF00059', 'RF00100', 'RF00163', 'RF00174', 'RF00177', 'RF00230',
    'RF00906', 'RF00994', 'RF01960', 'RF02271', 'RF04021', 'RF04088',
)
RNA_CLASSES = ('5S_rRNA', '5_8S_rRNA', 'CD-box', 'HACA-box', 'IRES', 'Intron_gpI',
               'Intron_gpII', 'leader', 'miRNA', 'riboswitch', 'ribozyme', 'scaRNA', 'tRNA')
POCKET_FEATURES = ('pkt_node_xyz', 'pkt_node_sca', 'pkt_node_vec', 'pkt_seq', 'ligand_center_of_mass')


def checked_feature(base, row):
    inputs = json.loads(row['inputs']) if isinstance(row['inputs'], str) else row['inputs']
    path = Path(base)/digest(row['sample_id'])/'features.npz'
    if sha256(path) != inputs['feature_sha256']:
        raise ValueError('Prepared input feature changed: ' + row['sample_id'])
    return path


def export_features(evidence, db, destination):
    for row in db.execute("SELECT sample_id,inputs FROM reference.samples WHERE split='T'"):
        source = checked_feature(evidence.package/'organizer/features', row)
        target = Path(destination)/'files'/digest(row['sample_id'])/'features.npz'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            checked_feature(Path(destination)/'files', row)
        else:
            shutil.copyfile(source, target)
            target.chmod(0o444)


def copy_resources(package, destination, manifest, *, deadline=None):
    """Copy only the hashed public resource allowlist into a worker input root."""
    from benchmarks.naturebench_active.preparation import check, file_hash, copy_file
    check(deadline)
    paths = {}
    for name, expected in manifest.get('reference_files', {}).items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid public resource path')
        source = Path(package)/'problem/reference'/relative
        if file_hash(source,deadline) != expected:
            raise ValueError('Public resource changed: ' + name)
        target = Path(destination)/'reference'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_hash(target,deadline) != expected:
                raise ValueError('Worker resource changed: ' + name)
        else:
            copy_file(source, target, deadline)
            target.chmod(0o444)
        paths[name] = str(target)
    if paths:
        (Path(destination)/'resources.json').write_text(json.dumps({
            'files': manifest['reference_files']}, sort_keys=True))
    return paths


def load_resource_paths(directory):
    index = Path(directory)/'resources.json'
    if not index.exists():
        return {}
    paths = {}
    for name, expected in json.loads(index.read_text())['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid released resource path')
        path = Path(directory)/'reference'/relative
        if sha256(path) != expected:
            raise ValueError('Released resource changed: ' + name)
        paths[name] = str(path)
    return paths

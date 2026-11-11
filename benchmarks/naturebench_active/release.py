"""Freeze and verify the tested active adapter and its actual data contracts."""
import json
from pathlib import Path
import platform
import subprocess
import sys

from rsi.common import digest, write_json
from rsi.freeze import code_hashes
from benchmarks.naturebench_active.common import sha256


def protocol_settings(config):
    from benchmarks.naturebench_active.timing import phase_plan
    from benchmarks.naturebench_active.validation_design import validate_policy
    policy=validate_policy(config.get('internal_validation'))
    return dict(internal_validation=policy,evaluation_phases=phase_plan(config.get('evaluation_phases')))


def contracts(config,task):
    package=Path(config['data_root'])/task;source=Path(config['naturebench_source_root'])/task
    metadata=json.loads((package/'metadata.json').read_text())
    from benchmarks.naturebench_active.adapters.multi_properties import check_package
    check_package(metadata)
    if sha256(package/'organizer/reference.sqlite')!=metadata['reference_sha256']:raise ValueError('Organizer reference changed')
    if sha256(package/'problem/data/samples.sqlite')!=metadata['public_sha256']:raise ValueError('Public data changed')
    paths=[package/'metadata.json',*(package/'problem').glob('*.md'),source/'metadata.json',source/'evaluation/evaluator.py']
    paths += [p for p in (source/'evaluation/shared').rglob('*') if p.is_file()]
    for name, expected in metadata.get('reference_files',{}).items():
        path=package/'problem/reference'/name
        if sha256(path)!=expected:raise ValueError('Public task resource changed')
        paths.append(path)
    if task=='CHEM_06':
        paths.append(source/'evaluation/qvina02')
    return {str(p.resolve()):sha256(p) for p in paths}


def release(config,task):
    frozen=json.loads(Path(config['freeze_manifest']).read_text())
    identity=frozen.pop('identity')
    if digest(frozen)!=identity:raise ValueError('Active freeze manifest changed')
    code=code_hashes()
    if frozen['code']!=code:raise ValueError('Active infrastructure differs from tested freeze')
    if frozen.get('protocol_settings')!=protocol_settings(config):raise ValueError('Active validation or evaluation phase protocol differs from tested freeze')
    if frozen['tasks'].get(task)!=contracts(config,task):raise ValueError('Active task differs from tested freeze')
    packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
    if frozen['python']!=sys.executable or frozen['packages']!=packages:raise ValueError('Active science environment changed')
    return dict(identity=identity,code=code,task_contract=frozen['tasks'][task],python=sys.executable,
                model=config['model'],protocol='naturebench-active',baseline=identity,
                protocol_settings=protocol_settings(config))


def freeze(config,evidence,manifest,tasks):
    report=json.loads(Path(evidence).read_text());code=code_hashes()
    if report.get('success') is not True or report.get('code_hashes')!=code:raise ValueError('Require passing tests for current exact infrastructure')
    settings=protocol_settings(config)
    if any(key in config for key in ('internal_validation','evaluation_phases')) and report.get('protocol_settings')!=settings:
        raise ValueError('Require test evidence binding the configured validation and phase protocol')
    manifest=Path(manifest)
    if manifest.exists():raise FileExistsError('Do not overwrite an existing experiment freeze')
    value=dict(protocol='naturebench-active',protocol_settings=settings,code=code,tasks={task:contracts(config,task) for task in tasks},
               python=sys.executable,python_version=platform.python_version(),
               packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),
               evidence=str(Path(evidence).resolve()),evidence_sha256=sha256(evidence))
    value['identity']=digest(value);write_json(manifest,value);manifest.chmod(0o444)
    return value['identity']

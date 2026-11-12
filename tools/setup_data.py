#!/usr/bin/env python3
"""Bind the unpacked RSI4S task_48 dataset to this checkout. Python 3.10+, standard library only.

  python tools/setup_data.py --bundle /data/rsi4s/rsi4s_task48 --out /data/rsi4s/bound \\
      --credentials ~/.config/rsi4s/credentials.json --python /envs/rsi4s/bin/python

The dataset itself is never modified and nothing large is copied. --out (new or empty) receives:
  packages/<TASK>/  task packages: links into the dataset, except organizer/source_root.txt -> out/sources
  sources/<TASK>/   evaluator and source resources, plus original-ID aliases used by hash-locked references
  medrsee_home/     MEDRSEE_HOME (data/medrsee -> the dataset's medrsee-v4 directory)
  state/            writable caches (ENV_02 prepared Test inputs)
  env.sh, setup_report.json
Then every group config listed in configs/task_48.json is updated in place: data roots, the rsi4s interpreter
(every group and both simulators use it), credentials and the simulator data. Rerunning updates the same keys.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.task_names import ORIGINAL_IDS  # noqa: E402  (standard library only)
from rsi.environment import mismatches  # noqa: E402  (standard library only)

SIMULATORS = ('MAT_01', 'CHEM_01')


def die(message):
    print('error: ' + message, file=sys.stderr)
    raise SystemExit(2)


def link(target, where):
    where.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, where)


def mirror(source, destination, overrides, relative=''):
    """Real directories only where a file must differ; everything else links into the dataset."""
    destination.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        name = f'{relative}{entry.name}'
        if name in overrides:
            continue
        if entry.is_dir() and any(o.startswith(name + '/') for o in overrides):
            mirror(entry, destination / entry.name, overrides, name + '/')
        else:
            link(entry.resolve(), destination / entry.name)


def interpreter(value, flag):
    path = Path(value).expanduser().absolute()  # keep a venv's bin/python link: it selects the venv
    if not path.is_file() or not os.access(path, os.X_OK):
        die(f'{flag}: {path} is not an executable Python interpreter')
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', type=Path, required=True, help='the unpacked rsi4s_task48 directory')
    parser.add_argument('--out', type=Path, required=True, help='new (or empty) directory for the binding')
    parser.add_argument('--credentials', type=Path, help='model API credentials file (JSON or CSV)')
    parser.add_argument('--python', required=True, help='interpreter of the rsi4s environment (requirements.txt)')
    args = parser.parse_args()

    bundle = args.bundle.expanduser().resolve()
    out = args.out.expanduser().absolute()
    if not (bundle / 'packages').is_dir() and not (bundle / 'medrsee-v4').is_dir() \
            and not (bundle / 'simulators').is_dir():
        die(f'{bundle} is not an unpacked rsi4s_task48 directory (run the dataset\'s tools/unpack.py first)')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        die(f'{out} is not an empty directory; choose a new one')
    if out == bundle or out.is_relative_to(bundle):
        die('--out must be outside the dataset directory')
    credentials = None
    if args.credentials:
        credentials = args.credentials.expanduser().resolve()
        if not credentials.is_file():
            die(f'--credentials: {credentials} does not exist')
    python = interpreter(args.python, '--python')

    out.mkdir(parents=True, exist_ok=True)
    report = dict(bundle=str(bundle), out=str(out), python=python, packages=[], sources=[], warnings=[])
    problems = mismatches(python)
    if problems:
        report['warnings'].append(f'--python does not match requirements.txt ({len(problems)} differences, e.g. '
                                  f'{"; ".join(problems[:3])}); prepare will refuse it until it does')
    packages, sources = out / 'packages', out / 'sources'
    packages.mkdir()
    sources.mkdir()
    for task_dir in sorted(p for p in (bundle / 'packages').glob('*') if p.is_dir()):
        mirror(task_dir, packages / task_dir.name, {'organizer/source_root.txt'})
        if (task_dir / 'organizer').is_dir():
            (packages / task_dir.name / 'organizer/source_root.txt').write_text(str(sources) + '\n')
        report['packages'].append(task_dir.name)
    for source_dir in sorted(p for p in (bundle / 'sources').glob('*') if p.is_dir()):
        link(source_dir.resolve(), sources / source_dir.name)
        if source_dir.name in ORIGINAL_IDS:  # hash-locked references name some source files by original ID
            link(source_dir.resolve(), sources / ORIGINAL_IDS[source_dir.name])
        report['sources'].append(source_dir.name)

    medrsee = bundle / 'medrsee-v4'
    medrsee_home = out / 'medrsee_home'
    if medrsee.is_dir():
        link(medrsee.resolve(), medrsee_home / 'data/medrsee')
    else:
        report['warnings'].append('No medrsee-v4/ in the dataset: MedRSEE tasks are not available')
    (out / 'state/medrsee-test-input-cache').mkdir(parents=True)
    simulators = {task: bundle / 'simulators' / task for task in SIMULATORS}
    for task, path in simulators.items():
        if not path.is_dir():
            report['warnings'].append(f'No simulators/{task} in the dataset: {task} is not available')

    index = json.loads((ROOT / 'configs/task_48.json').read_text())
    report['configs'] = {}
    for group in index['groups']:
        path = ROOT / group['config']
        config = json.loads(path.read_text())
        if config.get('naturebench_active'):
            config['data_root'] = str(packages)
            config['naturebench_source_root'] = str(sources)
        else:
            config['data_root'] = str(out)
        if 'medrsee_data_root' in config:
            config['medrsee_data_root'] = str(medrsee.resolve())
            config['medrsee_test_cache_root'] = str(out / 'state/medrsee-test-input-cache')
        if 'simulator_environments' in config:
            for task, environment in config['simulator_environments'].items():
                environment['data_root'] = str(simulators[task].resolve())
                environment['python'] = python
        config['python'] = python
        if credentials:
            config['credentials'] = str(credentials)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        report['configs'][group['config']] = dict(python=config['python'], data_root=config['data_root'])
    if not credentials:
        report['warnings'].append('--credentials not given: configs keep their credentials path')

    (out / 'env.sh').write_text(f'export MEDRSEE_HOME={medrsee_home}\nexport PYTHONDONTWRITEBYTECODE=1\n')
    (out / 'setup_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(out=str(out), packages=len(report['packages']), sources=len(report['sources']),
                          configs=sorted(report['configs']), warnings=report['warnings']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

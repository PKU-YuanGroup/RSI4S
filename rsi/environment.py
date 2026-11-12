"""The rsi4s environment: one interpreter for every task group, both simulators and the batch runner.

An interpreter qualifies when it is Python PYTHON_RELEASE and the distributions it actually imports are exactly
those pinned in requirements.txt (packaging tools aside). Extra packages are refused because they would change
which libraries the agents can use.
"""
import json
from pathlib import Path
import re
import subprocess

REQUIREMENTS = Path(__file__).resolve().parents[1] / 'requirements.txt'
PYTHON_RELEASE = '3.11.16'
# Packaging tools vary with how an environment was created; no task, evaluator or simulator imports them.
PACKAGING_TOOLS = frozenset({'pip', 'setuptools', 'wheel'})
# The first distribution of each name on sys.path is the one that imports (a venv shadows its base).
_PROBE = ('import importlib.metadata as m, json, platform\n'
          'seen = {}\n'
          'for d in m.distributions():\n'
          '    seen.setdefault(d.metadata["Name"], d.version)\n'
          'print(json.dumps(dict(python=platform.python_version(), packages=seen)))')


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def required():
    pins = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line and not line.startswith('-'):
            name, version = line.split('==')
            pins[canonical(name)] = version.strip()
    return pins


def installed(python):
    observed = json.loads(subprocess.check_output([str(python), '-I', '-c', _PROBE], text=True, timeout=120))
    packages = {}
    for name, version in observed['packages'].items():
        packages.setdefault(canonical(name), version)
    return observed['python'], packages


def mismatches(python):
    release, packages = installed(python)
    packages = {name: version for name, version in packages.items() if name not in PACKAGING_TOOLS}
    pins = required()
    problems = [] if release == PYTHON_RELEASE else [f'Python {release} (requires {PYTHON_RELEASE})']
    problems += [f'{name} {packages.get(name, "missing")} (requires {version})'
                 for name, version in sorted(pins.items()) if packages.get(name) != version]
    problems += [f'{name} {packages[name]} (not in requirements.txt)' for name in sorted(set(packages) - set(pins))]
    return problems


def verify(python):
    """Raise unless `python` is the rsi4s environment; return its description for readiness records."""
    problems = mismatches(python)
    if problems:
        shown = '; '.join(problems[:12]) + (f'; ... {len(problems) - 12} more' if len(problems) > 12 else '')
        raise ValueError(f'{python} is not the rsi4s environment of requirements.txt: {shown}')
    return dict(environment='rsi4s', python=str(python), python_version=PYTHON_RELEASE, packages=len(required()))

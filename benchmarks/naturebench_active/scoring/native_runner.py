"""Native scoring with NumPy pickle compatibility and an unchanged upstream runner.

Usage: python native_runner.py <native evaluator.py>; OUTPUT_DIR is required.
Only organizer-owned evaluation processes use this adapter.
"""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    # Published essentiality truth references NumPy 2's numeric module. NumPy
    # 1.26 provides the same array reconstruction function under numpy.core.
    try:
        importlib.import_module('numpy._core.numeric')
    except ModuleNotFoundError as exc:
        if exc.name not in ('numpy._core', 'numpy._core.numeric'):
            raise
        sys.modules.setdefault('numpy._core.numeric', importlib.import_module('numpy.core.numeric'))
    root = Path(__file__).resolve().parents[3]
    options = json.loads(os.environ.get('RSI_NATIVE_OPTIONS', '{}'))
    if options.get('task') in ('s41592-023-02148-8', 'CHEM_06', 'CHEM_05'):
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))
        spec = importlib.util.spec_from_file_location('native_terminal_evaluator', sys.argv[1])
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        if options['task'] == 's41592-023-02148-8':
            from benchmarks.naturebench_active.scoring.rna_terminal import run_evaluation
            metrics = run_evaluation(native, **options['rna'])
        elif options['task'] == 'CHEM_06':
            from benchmarks.naturebench_active.scoring.pocket_terminal import run_evaluation
            metrics = run_evaluation(native, **options['pocket'])
        else:
            from benchmarks.naturebench_active.scoring.reaction_terminal import run_evaluation
            metrics = run_evaluation(native)
        print('===EVAL_RESULT_JSON===\n'+json.dumps(metrics, allow_nan=False))
        return
    runner = root/'_evaluator_runner.py'
    sys.argv[0] = str(runner)
    runpy.run_path(str(runner), run_name='__main__')


if __name__ == '__main__':
    main()

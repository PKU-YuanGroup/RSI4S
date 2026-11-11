"""NatureBench-shaped adapter used by RSI4S's existing scoring service."""
import os
from pathlib import Path
from .scoring import score


def run_evaluation():
    task = os.environ['MEDRSEE_TASK']
    # This entry is exclusively validation: no environment-controlled switch to T.
    result = score(task, Path(os.environ['OUTPUT_DIR'])/'validation',
                   os.environ['MEDRSEE_INDEX'], os.environ['MEDRSEE_DATA'],
                   pool=os.environ['MEDRSEE_POOL'], scratch=Path.cwd()/'luna_validation')
    return {task:{result['metric']:result['score'], 'details':result,
                  'evaluation_split':result['split']}}

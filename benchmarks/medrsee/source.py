"""Shared paths and the fixed MedRSEE label contract. No split regeneration."""
import os
from pathlib import Path

WORK = Path(__file__).resolve().parent
# Defaults only: run configs pass medrsee_data_root explicitly. MEDRSEE_HOME/data/medrsee is the raw data.
ROOT = Path(os.environ.get('MEDRSEE_HOME', WORK.parents[1]))
DATA = ROOT / 'data/medrsee'
from .native_reader import iter_samples, NativeReader, SOURCE_TEMPLATES

# N, D0, additional labels, V, T; from the delivered split table.
COUNTS = {
 'RS_02': (1433,143,716,642,728), 'RS_01': (3799,379,1899,245,800),
 'ENV_01': (4412,441,2206,0,718), 'ENV_02': (7347592,734759,3673796,0,2409999),
 'ENV_03': (7000,700,3500,0,310), 'ENV_04': (140002,14000,70001,0,16025),
 'ENV_05': (53823,5382,26911,0,5110), 'MED_04': (800,80,400,0,88),
 'MED_05': (4537,453,2268,0,444), 'MED_01': (100,10,50,0,50),
 'MED_03': (118,11,59,0,13), 'MED_07': (80,8,40,40,40),
 'MED_08': (1499,149,749,0,376), 'MED_06': (95,9,47,0,10),
 'MED_02': (1000,100,500,0,251),
}

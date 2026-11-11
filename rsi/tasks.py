"""Resolve NatureBench, MedRSEE and fixed-scenario interactive task packages."""
from pathlib import Path
from rsi.common import ROOT

MEDRSEE_TASKS = ('RS_02','RS_01','ENV_01','ENV_02','ENV_03','ENV_04','ENV_05',
                 'MED_04','MED_05','MED_01','MED_03','MED_07','MED_08','MED_06','MED_02')
SIMULATOR_TASKS = ('MAT_01', 'CHEM_01')
RESP_TASKS = ('LIFE_01',)


def task_directory(data_root, task):
    if task in SIMULATOR_TASKS:
        return ROOT/'tasks/simulator'/task
    if task in MEDRSEE_TASKS:
        return ROOT/'tasks/medrsee'/task
    if task in RESP_TASKS and (data_root is None or not (Path(data_root)/task/'metadata.json').is_file()):
        return ROOT/'tasks/resp'/task
    return Path(data_root)/task


def selected_tasks():
    return [(t) for name in ('tasks_44.txt','tasks_medrsee.txt','tasks_simulator.txt','tasks_resp.txt')
            for t in (ROOT/'configs'/name).read_text().split()]

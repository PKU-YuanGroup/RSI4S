"""Task names and their original IDs.

Code, configs and task directories use the public names. Original IDs embed paper
DOIs, so they appear only here, in docs/task_48.md and at data boundaries: raw
NatureBench directories, hash-locked reference databases, splits.csv and old runs.
Names follow the order of docs/task_48.md.
"""

PUBLIC_NAMES = {
    's41467-023-36028-8': 'LIFE_01', 's41467-025-63412-3': 'LIFE_02', 's41467-025-64512-w': 'LIFE_03',
    's41551-024-01312-5': 'LIFE_04', 's41587-024-02414-w': 'LIFE_05', 's41587-024-02428-4': 'LIFE_06',
    's41592-023-02032-5': 'LIFE_07', 's41592-024-02372-w': 'LIFE_08', 's41592-024-02523-z': 'LIFE_09',
    's41592-025-02776-2': 'LIFE_10', 's42256-024-00836-4': 'LIFE_11', 's42256-025-01026-6': 'LIFE_12',
    's42256-025-01042-6': 'LIFE_13', 's43588-024-00716-2': 'LIFE_14', 's43588-024-00733-1': 'LIFE_15',
    's43588-025-00878-7': 'LIFE_16',
    'EXP-07': 'CHEM_01', 's42256-022-00447-x': 'CHEM_02', 's42256-022-00501-8': 'CHEM_03',
    's42256-023-00636-2': 'CHEM_04', 's42256-023-00639-z': 'CHEM_05', 's42256-023-00712-7': 'CHEM_06',
    's42256-024-00815-9': 'CHEM_07', 's42256-024-00916-5': 'CHEM_08', 's42256-024-00956-x': 'CHEM_09',
    's42256-025-01010-0': 'CHEM_10', 's43588-024-00757-7': 'CHEM_11',
    'EXP-01': 'MAT_01', 's42256-023-00628-2': 'MAT_02', 's43588-025-00842-5': 'MAT_03',
    's43588-025-00903-9': 'MAT_04', 's43588-025-00917-3': 'MAT_05',
    'RS-07': 'RS_01', 'RS-18': 'RS_02',
    'EE-01': 'ENV_01', 'EE-02': 'ENV_02', 'EE-03': 'ENV_03', 'EE-04': 'ENV_04', 'EE-05': 'ENV_05',
    'MED-03': 'MED_01', 'MED-04': 'MED_02', 'MED-05': 'MED_03', 'MED-06': 'MED_04', 'MED-11': 'MED_05',
    'MED-14': 'MED_06', 'MED-19': 'MED_07', 'MED-20': 'MED_08', 's43588-025-00798-6': 'MED_09',
}
ORIGINAL_IDS = {public: original for original, public in PUBLIC_NAMES.items()}


def public_name(task):
    """Old records carry original IDs; fixtures and other tasks keep their own name."""
    return PUBLIC_NAMES.get(task, task)


def original_id(task):
    return ORIGINAL_IDS.get(task, task)

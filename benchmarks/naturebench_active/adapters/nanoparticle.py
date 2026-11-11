"""Public nanoparticle input contract and synthetic shape checks.

These examples are constructed from the published schema. They contain no
dataset records, labels, or independent Test information.
"""

TASK = 'MAT_05'

OUTPUT_CONTRACT = '''Task-specific override: output exactly {"target": finite scalar}. The training annotation is a full spectrum, but the scalar scientific target is log10(100 + I), where I=sum(wavelength_spectrum_y[j] for j with 300 < abs(wavelength_spectrum_x[j]) < 450). Derive this from acquired training labels only. Predict from radii, dopant types/concentrations, shell geometries and structural fields; never use _id as a feature. No spectrum is present at prediction time. The published task includes 0–3 shells during training and 4 shells in OOD: the core is also a region, so predict must support 1–5 regions. Do not infer a maximum of four regions from training data. Keep feature width fixed across fit and predict, preserve the fifth region (for example with five padded slots and masks, or variable-length aggregation), and compute outer-shell/volume features from the actual region count. dopant_concentration and formula_by_constraint are top-level inputs; constraints, dopant_specifications and n_levels are inside inputs["input"]. After fitting, the host checks prediction on synthetic unlabeled 1–5-region examples built only from this public schema. These are interface checks, not validation scores or Test data. The native metric is MSE*1000 on this log target. An internal D0 holdout is not the independent IID/OOD Test evaluation.'''


def schema_probes():
    records = []
    for regions in range(1, 6):
        concentrations = [{'Yb': .2, 'Er': .02} for _ in range(regions)]
        inputs = dict(
            _id={'@class': 'ObjectId', '@module': 'bson.objectid',
                 'oid': f'{regions:024x}'},
            input=dict(
                constraints=[{'@class': 'SphericalConstraint',
                              '@module': 'NanoParticleTools.inputs.nanoparticle',
                              '@version': None, 'host_structure': None,
                              'radius': float(20 + 10 * i)} for i in range(regions)],
                dopant_specifications=[[i, amount, element, 'Y']
                                       for i, composition in enumerate(concentrations)
                                       for element, amount in composition.items()],
                n_levels=[2, 24, 34]),
            dopant_concentration=concentrations,
            formula_by_constraint=['Yb20Er2' for _ in range(regions)])
        records.append(dict(sample_id=f'public-schema/regions-{regions}',
                            instance='sunset_1', inputs=inputs))
    return records

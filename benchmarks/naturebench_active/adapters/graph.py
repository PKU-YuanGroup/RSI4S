"""Label-free features from the explicitly public development graph assets."""
import json
import numpy as np
from scipy.sparse import coo_matrix

from benchmarks.naturebench_active.common import connect


def network_features(pool, *, scope='development', deadline=None):
    from benchmarks.naturebench_active.preparation import check, database
    check(deadline)
    result={}
    with database(pool,deadline) as db:
        for row in db.execute('SELECT asset_id,payload FROM assets ORDER BY asset_id'):
            if not row['asset_id'].endswith('/'+scope):continue
            check(deadline)
            asset=json.loads(row['payload'])
            network=row['asset_id'].split('/')[1]
            node_ids=np.asarray(asset['node_indices'],dtype=np.int64)
            edges=np.asarray(asset['edge_index'],dtype=np.int64)
            features=np.asarray(asset['features'],float)
            if features.shape!=(len(node_ids),64):raise ValueError('Unexpected graph features')
            if len(set(node_ids))!=len(node_ids):raise ValueError('Duplicate native node index')
            lookup={int(node):i for i,node in enumerate(node_ids)}
            src=np.fromiter((lookup[int(x)] for x in edges[0]),dtype=np.int64)
            dst=np.fromiter((lookup[int(x)] for x in edges[1]),dtype=np.int64)
            graph=coo_matrix((np.ones(len(src)),(src,dst)),shape=(len(node_ids),len(node_ids))).tocsr()
            graph=graph.maximum(graph.T)
            graph.setdiag(0);graph.eliminate_zeros();graph.data[:]=1
            degree=np.asarray(graph.sum(axis=1)).ravel()
            neighbor=(graph@features)/np.maximum(degree[:,None],1)
            for i,gene in enumerate(asset['gene_names']):
                check(deadline)
                result.setdefault(gene,{})[network]=dict(features=features[i].tolist(),
                    neighbor_mean=neighbor[i].tolist(),degree=int(degree[i]))
    return result


def enrich(records, features, *, deadline=None):
    from benchmarks.naturebench_active.preparation import check
    for row in records:
        check(deadline)
        value=features[row['inputs']['gene']]
        if row['inputs']['network'] not in value:raise ValueError('Gene absent from its native graph')
        row['inputs']['network_features']=value

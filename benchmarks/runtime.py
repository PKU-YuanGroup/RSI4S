"""One benchmark entry point and shared active-evidence capabilities."""
import time

from rsi.evaluation import Evaluation
from rsi.tasks import MEDRSEE_TASKS, SIMULATOR_TASKS
from rsi.schedule import final_inference_seconds
from benchmarks.evidence import TASK_OPS, META_OPS


def create_evaluation(config, task, data_dir, directory, gpu, journal):
    args = (task, data_dir, directory, config['budget']['solve_seconds'], gpu,
            config['budget'].get('wall_seconds'), config.get('sandbox_resources'))
    if config.get('naturebench_active'):
        from benchmarks.naturebench_active.runtime import ActiveEvaluation
        return ActiveEvaluation(*args, source_root=config['naturebench_source_root'], journal=journal,
                                validation_policy=config.get('internal_validation'),
                                evaluation_phases=config.get('evaluation_phases'),
                                final_inference_seconds=final_inference_seconds(config))
    if task in SIMULATOR_TASKS:
        from benchmarks.simulator.runtime import SimulatorEvaluation
        return SimulatorEvaluation(*args, journal=journal,
            **config.get('simulator_environments', {}).get(task, {}))
    if task in MEDRSEE_TASKS:
        from benchmarks.medrsee.runtime import MedRSEEEvaluation
        from benchmarks.medrsee.source import DATA
        return MedRSEEEvaluation(*args, journal=journal,
            raw_data=config.get('medrsee_data_root', str(DATA)),
            cache_root=config.get('medrsee_cache_root'), test_cache_root=config.get('medrsee_test_cache_root'),
            index_root=config.get('medrsee_index_root'))
    return Evaluation(*args)


class EvidenceEvaluation(Evaluation):
    """Adapters supply data access, validation rules and submission eligibility."""
    annotation_protocol = None
    closed = False

    def annotation_status(self):
        status = self.evidence.status()
        if self.annotation_protocol:
            status['protocol'] = self.annotation_protocol
        return status

    def evidence_dispatch(self, op, request, kind, run_id, remaining, tool_call=0):
        allowed = {'task': TASK_OPS, 'meta': META_OPS}.get(kind, set())
        if op not in allowed:
            raise PermissionError(f'Capability denied for {kind}: {op}')
        if self.closed:
            raise ValueError('Research is closed; evidence tools are unavailable')
        if remaining <= 0:
            raise TimeoutError('Evidence operation deadline expired')
        ids = request.get('sample_ids')
        deadline = time.monotonic() + remaining
        if op == 'annotation_status':
            return self.annotation_status()
        if op == 'select_validation':
            return self.evidence.select_validation(ids, actor=kind, deadline=deadline)
        if op == 'query_labels':
            return self.evidence.query(ids, actor=kind, run_id=run_id,
                                       reason=request['reason'], deadline=deadline)
        if op == 'request_labels':
            return self.evidence.request(actor=kind, reason=request['reason'], sample_ids=ids,
                criteria=request.get('criteria'), count=request.get('count'), run_id=run_id, tool_call=tool_call)
        return self.evidence.fetch(request.get('split', 'train'), ids,
            request.get('offset', 0), request.get('limit', 256), deadline=deadline, pages=request.get('pages', 1))

    def selection_candidates(self, runs):
        return self.selection_module.selection_candidates(self.task, runs, self.state.primary_table)

    def select_submission(self, runs, decision):
        return self.selection_module.select_submission(self.task, runs, self.state.primary_table, decision)

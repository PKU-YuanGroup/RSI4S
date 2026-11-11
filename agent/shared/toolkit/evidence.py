"""Role-scoped tools shared by the active evidence protocols."""

def tools(role, protocol=None):
    ids = {'sample_ids': {'type':'array','items':{'type':'string'}},
           'sample_ids_file': {'type':'string','description':'Workspace-relative UTF-8 file, one sample ID per line.'}}
    shared = [
        {'name':'annotation_status','description':'Read current shared annotation budget and dataset/validation membership. No labels are charged.',
         'input_schema':{'type':'object','properties':{},'required':[]}},
        {'name':'fetch_data','description':'Fetch permitted inputs and acquired labels. Q inputs are available; hidden labels and T are unavailable. limit is 1..4096 per page; pages=1..256 fetches multiple pages without model calls and returns batches/next_offset. Completed batches are indexed in DATA_DIR/batches.json. Use reader.iter_batch to inspect a small sample; avoid repeating Task training in Meta.',
         'input_schema':{'type':'object','properties':{**ids,
             'split':{'type':'string','enum':['train','val','pool','labeled','D0','Q','V','all']},
             'offset':{'type':'integer'},'limit':{'type':'integer'},'pages':{'type':'integer'}},'required':[]}},
    ]
    if role == 'meta':
        for name, purpose in [('select_final_artifact', 'Record or replace your choice of an existing model during the read-only review after research closes.'),
                              ('finish_experiment', 'Select an existing model and end the read-only final review. Research has already closed; no successor is published.')]:
            shared.append({'name': name, 'description': purpose +
                ' Compare historical trajectories, verified validation scores and validation cohort hashes yourself. '
                'Choose an eligible artifact ID and explain the comparison. The host does not choose the highest score. '
                'No Test feedback is available. Only a completed verified artifact can be selected.',
                'input_schema': {'type': 'object', 'properties': {
                    'artifact': {'type': 'string'}, 'reason': {'type': 'string'}},
                    'required': ['artifact', 'reason']}})
        shared.append({'name':'query_labels','description':'Approve an expert annotation query for exact Q sample IDs. Charges each new sample once against the shared additional 50% budget. You may approve, modify or reject Task requests, or initiate a query yourself. Explain the missing evidence and selection rationale.',
          'input_schema':{'type':'object','properties':{**ids,'reason':{'type':'string'}},'required':['reason']}})
    else:
        shared += [
            {'name':'select_validation','description':'Register the exact acquired sample IDs to use for internal validation where official V is absent. Each call replaces the selection; [] clears it. Choose your own splitting tools and revise the selection during research. The host does not expand groups or dates, preview or diagnose the split. Official V stays unchanged. Read current roles through the public data interface. Pass sample_ids or a workspace sample_ids_file.',
             'input_schema':{'type':'object','properties':ids,'required':[]}},
            {'name':'request_labels','description':'Request additional data/labels by proposing sample IDs or an information gap to Meta. An accepted request immediately ends this Task stage and starts the next Meta stage within the shared research budget, even without an evaluated baseline. Save useful files and notes before calling. Meta decides whether, what, how many and when to annotate; this request itself charges no labels. Invalid requests return an error without handing off.',
             'input_schema':{'type':'object','properties':{**ids,'criteria':{'type':'string'},'count':{'type':'integer'},'reason':{'type':'string'}},'required':['reason']}},
        ]
    if protocol=='naturebench-active':
        descriptions={
            'fetch_data':'Read one JSON page of permitted records from samples.sqlite. Only acquired targets are returned; Q targets and Test are unavailable. limit 1..4096; use offset for pagination, pages must be 1. Returns records and sanitized attachment paths, not MedRSEE batches.',
            'query_labels':'Acquire exact Q sample IDs against the shared additional annotation budget. Each newly acquired unique record is charged once. Newly acquired Q labels remain training members until explicitly selected for validation. Explain the evidence gap and selection rationale; Test, official V and unknown IDs are rejected.',
        }
        for item in shared:
            if item['name'] in descriptions:item['description']=descriptions[item['name']]
    if role == 'task':
        descriptions = {None: {'fetch_data': 'Fetch permitted inputs and acquired labels. Q inputs are available; hidden labels and T are unavailable. limit is 1..4096 per page; pages=1..256 fetches multiple pages without model calls and returns batches/next_offset. All completed batches are indexed in DATA_DIR/batches.json. Use reader.iter_batch(role="train") or lazy load_batch; 3-D arrays may be read-only memory maps.'}, 'naturebench-active': {'fetch_data': 'Read one JSON page of permitted records from samples.sqlite, with acquired labels and optional sanitized attachment paths. Hidden Q labels and Test are unavailable. limit 1..4096; use offset for pagination. pages must be 1. Large arrays/mmCIF strings can be inspected directly in the public SQLite database; do not use MedRSEE readers.'}}
        for item in shared:
            if item['name'] in descriptions.get(protocol, {}):
                item['description'] = descriptions[protocol][item['name']]
    return shared

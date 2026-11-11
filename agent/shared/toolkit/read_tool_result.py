"""Read an exact host-recorded tool result without granting filesystem access."""


def tool_info():
    return {
        'name': 'read_tool_result',
        'description': 'Read an earlier tool result or archived conversation from this session by call_id. Conversation archives contain exact messages and a previous_archive link. Use field stdout for exact command output, or json for the executed request and raw response. Offsets count characters. Pages also have a 24 KiB JSON text byte cap; always follow returned next_offset rather than adding limit. Read only the relevant ranges.',
        'input_schema': {'type': 'object', 'properties': {
            'call_id': {'type': 'string', 'description': 'ID returned in _tool_result.call_id or history_archive.'},
            'field': {'type': 'string', 'enum': ['json', 'stdout']},
            'offset': {'type': 'integer', 'description': 'Zero-based character offset; default 0.'},
            'limit': {'type': 'integer', 'description': 'Characters to read, 1..50000; default 8000.'},
        }, 'required': ['call_id']},
    }

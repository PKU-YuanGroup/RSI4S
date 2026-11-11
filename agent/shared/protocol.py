# Adapted from facebookresearch/HyperAgents, commit 59a68f672dfb92c74aeb7e61535d776fb36e172d.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# CC BY-NC-SA 4.0; see HYPERAGENTS-LICENSE.md in the repository root.
"""HyperAgents text tool protocol, adapted to the resumable host bridge."""
import json
import hashlib
import re

DSML_PREFIX = r'[｜|]{1,2}DSML[｜|]{1,2}'

def get_tooluse_prompt(tool_infos=[]):
    """
    Get the prompt for using the available tools.
    """
    # If no tools are available, return an empty string
    if not tool_infos or len(tool_infos) == 0:
        return ""
    # Create the prompt
    tools_available = [str(tool_info) for tool_info in tool_infos]
    tools_available = '\n\n'.join(tools_available) if tools_available else 'None'
    tooluse_prompt = """Here are the available tools:
```
{tools_available}
```

Use only one tool (if needed) in this format:
<json>
{{
    "tool_name": ...,
    "tool_input": ...
}}
</json>

ONLY USE ONE TOOL PER RESPONSE, AND STRICTLY FOLLOW THE FORMAT OF TOOL_NAME AND TOOL_INPUT ABOVE.
DO NOT HALLUCINATE OR MAKE UP ANYTHING.
Results carry call_id and omit repeated inputs. Use read_tool_result to recover exact
output if the display is clipped or cleaned. Do not append budget snapshots to history.
""".format(tools_available=tools_available)
    return tooluse_prompt.strip()


def parse_dsml(text):
    """Adapt the provider's observed DSML wire format without altering tool input."""
    calls = list(re.finditer(r'<(' + DSML_PREFIX + r')\s+calls>(.*?)</\1\s+calls>', text, re.DOTALL))
    if not calls:
        raise ValueError('Incomplete DSML call; return one complete <json> tool call.')
    if len(calls) != 1:
        raise ValueError('Return only one tool call per response.')
    call = calls[0]
    prefix = re.escape(call[1])
    invocation = re.compile(r'<' + prefix + r'\s+invoke\s+name="([^"]+)">(.*?)</' + prefix + r'\s+invoke>', re.DOTALL)
    invoke = invocation.fullmatch(call[2].strip())
    if len(invocation.findall(call[2])) != 1:
        raise ValueError('Return exactly one complete DSML invocation.')
    if not invoke:
        raise ValueError('Malformed DSML invocation; use tool_name and tool_input.')
    parameter = re.compile(r'<' + prefix + r'\s+parameter\s+name="([^"]+)"\s+string="(true|false)">(.*?)</' + prefix + r'\s+parameter>', re.DOTALL)
    if parameter.sub('', invoke[2]).strip():
        raise ValueError('Malformed DSML parameters')
    arguments = {}
    for name, is_string, value in parameter.findall(invoke[2]):
        if name in arguments:
            raise ValueError('Duplicate DSML parameter: ' + name)
        arguments[name] = value if is_string == 'true' else json.loads(value)
    return {'tool_name': invoke[1], 'tool_input': arguments}


def recover_outer_tool_brace(text):
    """Recover only a single missing outer brace of one wrapped tool object.

    No command, string, nested object or final answer is repaired. The raw
    response stays in history; the action records the exact transport repair.
    Duplicate keys and additional actions remain errors.
    """
    stripped=text.strip()
    if not (stripped.startswith('<json>') and stripped.endswith('</json>')):
        return None
    payload=stripped[len('<json>'):-len('</json>')].strip()
    if not payload.startswith('{') or not payload.endswith('}'):
        return None
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    try:
        value=json.loads(payload+'}',object_pairs_hook=unique)
    except ValueError:
        return None
    if (not isinstance(value,dict) or set(value)!={'tool_name','tool_input'}
            or not isinstance(value['tool_name'],str) or not isinstance(value['tool_input'],dict)):
        return None
    value['_protocol_repair']={'kind':'missing_outer_object_brace_v1',
        'raw_response_sha256':hashlib.sha256(text.encode()).hexdigest()}
    return value


def parse_response(text):
    stripped=text.strip()
    if re.match(r'</?(?:bash|tool_call|tool_use|calls|invoke|parameter)(?=[\s>/]|$)', stripped):
        raise ValueError('Unsupported tool markup; return one <json> block with tool_name and object tool_input.')
    markers=('<json>','</json>','<｜DSML','<｜｜DSML','<|DSML','<||DSML')
    if stripped and any(marker.startswith(stripped) for marker in markers):
        raise ValueError('Incomplete protocol marker; return a complete <json> response or tool call.')
    native = re.search(r'<' + DSML_PREFIX, text)
    json_start = text.find('<json>')
    if text.lstrip().startswith('{'):
        json_start = len(text) - len(text.lstrip())
    if native and (json_start < 0 or native.start() < json_start):
        return parse_dsml(text)
    # Decode the JSON payload before looking for its closing wrapper. Literal
    # protocol tags inside a command are data and must not truncate the action.
    blocks = []
    if stripped.startswith('{') or json_start < 0:
        blocks = [text]
    else:
        cursor = json_start
        while cursor >= 0:
            begin = cursor + len('<json>')
            while begin < len(text) and text[begin].isspace():
                begin += 1
            try:
                _, end = json.JSONDecoder().raw_decode(text, begin)
            except json.JSONDecodeError:
                recovered=recover_outer_tool_brace(text)
                if recovered is not None:return recovered
                raise ValueError('Malformed tool JSON; use tool_name and object tool_input.') from None
            closing = re.match(r'\s*</json>', text[end:])
            if not closing:
                raise ValueError('Incomplete <json> response; return a complete JSON block.')
            blocks.append(text[begin:end])
            cursor = text.find('<json>', end + closing.end())
        if len(blocks) != 1:
            raise ValueError('Return only one tool call or final response per message.')
    for block in blocks:
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            if 'tool_name' in block or '<json>' in text:
                raise ValueError('Malformed tool JSON; use tool_name and object tool_input.')
            continue
        if isinstance(value, dict):
            if 'tool_name' in value or 'tool_input' in value:
                if ('tool_input' not in value and isinstance(value.get('tool_name'),str)
                        and len(value)>1 and not {'response','tool_calls','_protocol_repair'} & value.keys()):
                    # Observed provider JSON mode flattens arguments alongside
                    # tool_name. Re-nest exactly those fields; dispatch still
                    # validates every parameter against the available tool.
                    def unique(pairs):
                        result={}
                        for key,item in pairs:
                            if key in result:raise ValueError('Duplicate tool JSON key')
                            result[key]=item
                        return result
                    value=json.loads(block,object_pairs_hook=unique)
                    value={'tool_name':value.pop('tool_name'),'tool_input':value,
                        '_protocol_repair':{'kind':'flat_tool_arguments_v1',
                            'raw_response_sha256':hashlib.sha256(text.encode()).hexdigest()}}
                if not isinstance(value.get('tool_name'), str) or not isinstance(value.get('tool_input'), dict):
                    raise ValueError('A tool call requires string tool_name and object tool_input.')
                return value
            if 'response' in value:
                return {'response': value['response']}
    # Corrupted provider tags are an attempted action, not a scientific final
    # answer. Let the same Agent repair its format without handing off a stage.
    if re.search(r'<[^<>\n]*DSML[^<>\n]*(?:>|$)', text):
        raise ValueError('Malformed DSML tool call; return one complete <json> tool call.')
    if 'tool_name' in text or 'tool_input' in text:
        raise ValueError('Malformed tool call; return one complete <json> block.')
    if not stripped or re.fullmatch(r'(?:<[^<>\n]+>\s*)+', stripped):
        raise ValueError('Empty or markup-only output is not a handoff. Return a complete tool call '
                         'or a <json> response with your results or failure evidence.')
    return {'response': text}


def dispatch(action, tool_infos):
    name, arguments = action['tool_name'], action['tool_input']
    info = next((item for item in tool_infos if item['name'] == name), None)
    if info is None:
        raise ValueError('Unknown tool: ' + name)
    schema = info['input_schema']
    # Some providers spell omitted optional parameters as explicit nulls.
    arguments = {key: value for key, value in arguments.items()
                 if value is not None or key in schema.get('required', [])}
    if set(schema.get('required', [])) - arguments.keys():
        raise ValueError('Missing required tool parameters')
    if arguments.keys() - schema['properties'].keys():
        raise ValueError('Unknown tool parameters')
    types = {'string': str, 'integer': int, 'number': (int, float), 'array': list, 'object': dict}
    for key, value in arguments.items():
        field = schema['properties'][key]
        if not isinstance(value, types[field['type']]) or isinstance(value, bool):
            raise ValueError(f"Wrong type for {key}: expected {field['type']}")
        if 'enum' in field and value not in field['enum']:
            raise ValueError('Invalid value for ' + key)
        if field['type'] == 'array' and 'items' in field:
            if any(type(item) is not types[field['items']['type']] for item in value):
                raise ValueError('Wrong item type for ' + key)
    return {'op': name, **arguments}

"""Explicit provider contracts; credentials stay outside persisted study records."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from urllib.parse import urlsplit

GLM_MODEL = 'ZHIPU/GLM-5.3-Flash'
MIMO_MODEL = 'mimo-v2.6-pro'
# DeepSeek V4.1 Flash served by Alibaba Bailian's OpenAI-compatible endpoint, using the Bailian credential.
DS_BAILIAN_MODEL = 'deepseek-v4.1-flash-1017'
# Gemini 4 Argon through an OpenAI-compatible endpoint (Google's, or a gateway); the credential JSON gives the base URL.
GEMINI_ARGON_MODEL = 'gemini-4-argon'
# Kimi K3.1 Pro on Moonshot's OpenAI-compatible Kimi API (https://api.moonshot.ai/v1 or https://api.moonshot.cn/v1).
KIMI_MODEL = 'kimi-k3.1-pro'
SUPPORTED_MODELS = ('deepseek-flash', 'qwen3.8-flash', GLM_MODEL, MIMO_MODEL, DS_BAILIAN_MODEL, GEMINI_ARGON_MODEL,
                    KIMI_MODEL)
QWEN_MODEL_SOURCE = 'https://help.aliyun.com/zh/model-studio/qwen3-8-flash'
QWEN_API_SOURCE = 'https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions/'
GLM_MODEL_SOURCE = 'https://help.aliyun.com/en/model-studio/glm-5-3-flash-by-zhipu'
GLM_API_SOURCE = 'https://help.aliyun.com/zh/model-studio/glm-zhipu'
DEEPSEEK_API_SOURCE = 'https://api-docs.deepseek.com/guides/thinking_mode/'
MIMO_API_SOURCE = 'https://mimo.mi.com/docs/zh-CN/api/chat/openai-api'
MIMO_MODEL_SOURCE = 'https://mimo.mi.com/docs/en-US/quick-start/model'
GEMINI_MODEL_SOURCE = 'https://ai.google.dev/gemini-api/docs/models/gemini-4-argon'
GEMINI_API_SOURCE = 'https://ai.google.dev/gemini-api/docs/latest-model'
KIMI_API_SOURCE = 'https://platform.kimi.ai/docs/api/chat.md'
KIMI_MODEL_SOURCE = 'https://platform.kimi.ai/docs/api/models-overview.md'
KIMI_THINKING_SOURCE = 'https://platform.kimi.ai/docs/guide/use-thinking-models.md'
COMPLETE_HISTORY_MAX_OUTPUT = 16384
# transport_v2: every model streams. A request fails after this many seconds
# without a token. The absolute limit only applies to responses that are not SSE.
TRANSPORT_PROTOCOL = 'transport_v2'
TOKEN_STALL_SECONDS = 180
API_REQUEST_TIMEOUT_SECONDS = 600

# Translate the shared policy into supported API fields. Experimental limits
# belong in generation_settings, never in these per-provider contracts.
# Alibaba Bailian (Qwen, GLM, DeepSeek on Bailian) moderates output while it streams. Observed 2026-11-06
# (DS fixed/LIFE_03, a Meta call): 186 ordinary text chunks, then an in-stream error with HTTP 200,
# code data_inspection_failed, "Output data may contain inappropriate content." It refuses this sample, not
# the request, so it is retried. Input inspection ("Input data ...") is not declared: never observed.
BAILIAN_BLOCKED_ERRORS = (dict(name='output_inspection', http_status=(200,),
                               messages=('Output data may contain inappropriate content',)),)

_API_CONTRACTS = {
    'deepseek-flash': dict(
        output_parameter='max_tokens', request_options=dict(thinking=dict(type='enabled')),
        source=DEEPSEEK_API_SOURCE, built_in_web_search='no search tools or options requested'),
    'qwen3.8-flash': dict(
        output_parameter='max_completion_tokens',
        request_options=dict(enable_thinking=True, preserve_thinking=False, enable_search=False),
        source=QWEN_API_SOURCE, built_in_web_search='explicitly disabled',
        blocked_errors=BAILIAN_BLOCKED_ERRORS),
    DS_BAILIAN_MODEL: dict(
        # Probed 2026-11-03: the endpoint validates max_completion_tokens (range [1, 393216]) but accepted
        # max_tokens=100000000 without a limit, so only max_completion_tokens bounds the output here.
        # `thinking.type` (the earlier DeepSeek form) is honored: 'disabled' removes the reasoning stream.
        output_parameter='max_completion_tokens',
        request_options=dict(thinking=dict(type='enabled')),
        source=QWEN_API_SOURCE, built_in_web_search='no search tools or options requested',
        blocked_errors=BAILIAN_BLOCKED_ERRORS),
    GLM_MODEL: dict(
        output_parameter='max_tokens',
        request_options=dict(thinking=dict(type='enabled', clear_thinking=True)),
        source=GLM_API_SOURCE, built_in_web_search='unsupported; no search options requested',
        documented_default_reasoning_effort='max',
        blocked_errors=BAILIAN_BLOCKED_ERRORS),
    MIMO_MODEL: dict(
        output_parameter='max_completion_tokens',
        # The text bridge accepts bare JSON as well as <json> wrappers. MiMo
        # otherwise sometimes emits repeated parallel XML calls despite the
        # single-call prompt. Constrain the wire output, keeping thinking on.
        request_options=dict(thinking=dict(type='enabled'),response_format=dict(type='json_object')),
        agent_json_protocol='single_tool_object_v1',
        source=MIMO_API_SOURCE, built_in_web_search='no search tools or options requested',
        provider_controls_sampling=True,
        documented_effective_sampling=dict(temperature=1.0, top_p=0.95)),
    GEMINI_ARGON_MODEL: dict(
        # Google documents thinking levels low/medium/high (default medium); reasoning_effort selects the highest.
        # The reference results used an OpenAI-compatible gateway, which accepted undocumented values such as max
        # or xhigh without error; they are not used.
        output_parameter='max_tokens',
        request_options=dict(reasoning_effort='high'),
        source=GEMINI_API_SOURCE,
        built_in_web_search='no search tools or options requested',
        documented_default_reasoning_effort='medium', documented_reasoning_levels=('low', 'medium', 'high'),
        # Google documents that this model ignores custom temperature/top_p/top_k.
        provider_controls_sampling=True,
        documented_effective_sampling=dict(temperature=1.0),
        # Seen in the reference runs: a safety refusal returned as ordinary assistant text with finish_reason=stop.
        # It refuses this sample, never answers the Agent, and replays of the same requests passed, so it is
        # retried. finish_reason is not used: a gateway labelled complete, valid answers content_filter.
        blocked_response=dict(content_prefixes=("This request was blocked by Gemini's filters",)),
        # Also seen: the whole request rejected with HTTP 400 before any output ("request blocked by (SAFETY|OTHER)",
        # or PROHIBITED_CONTENT) on ordinary modelling code; replays of the exact requests passed, so it is retried.
        # Other 400s still fail closed.
        blocked_errors=(dict(name='blocked_request', http_status=(400,),
                             messages=('request blocked by (', 'PROHIBITED_CONTENT',
                                       "This request was blocked by Gemini's filters")),)),
    KIMI_MODEL: dict(
        # Moonshot documents reasoning_effort low/high/max (default max) for the K3 series; thinking is always on
        # and the `thinking` parameter is not supported. max_completion_tokens replaces the deprecated max_tokens.
        output_parameter='max_completion_tokens',
        request_options=dict(reasoning_effort='max'),
        source=KIMI_API_SOURCE,
        built_in_web_search='no search tools or options requested',
        documented_default_reasoning_effort='max', documented_reasoning_levels=('low', 'high', 'max'),
        # The K3 series fixes temperature 1.0, top_p 0.95, n 1 and zero penalties; any other value is an error,
        # so none is sent.
        provider_controls_sampling=True,
        documented_effective_sampling=dict(temperature=1.0, top_p=0.95),
        # Moonshot asks multi-turn requests to pass earlier reasoning_content back ("Preserved Thinking"). Like
        # every model here, the harness replays role/content only; Agent turns are plain text, not native tool
        # calls. The documentation does not say what omitting it does.
        documented_history=('Preserved Thinking: earlier reasoning_content is requested in multi-turn history; '
                            'not replayed (uniform context policy); see ' + KIMI_THINKING_SOURCE)),
}

# Actual provider capacities are metadata, not experimental quotas. DeepSeek
# supplies its capacity in /models; GLM/Qwen currently need documented values.
_DOCUMENTED_LIMITS = {
    GLM_MODEL: dict(max_output_tokens=131072, context_window=1048576,
                    source=GLM_MODEL_SOURCE, verified_at='2026-10-27'),
    'qwen3.8-flash': dict(max_output_tokens=131072, context_window=1000000,
                        source=QWEN_MODEL_SOURCE, verified_at='2026-10-25'),
    DS_BAILIAN_MODEL: dict(max_output_tokens=393216, context_window=1048576,
                           source=QWEN_API_SOURCE, verified_at='2026-11-03',
                           capacity_note='Same capacity as the earlier official DeepSeek API configuration (user-specified); '
                                         'the endpoint reports max_completion_tokens range [1, 393216]; the context window '
                                         'is not served by the endpoint and is not independently verified.'),
    MIMO_MODEL: dict(max_output_tokens=131072, context_window=1000000,
                    source=MIMO_MODEL_SOURCE, output_source=MIMO_API_SOURCE,
                    context_window_note='Official 1M label represented conservatively as 1,000,000 tokens',
                    verified_at='2026-10-30'),
    GEMINI_ARGON_MODEL: dict(max_output_tokens=65536, context_window=1048576,
                             source=GEMINI_MODEL_SOURCE, verified_at='2026-11-07',
                             capacity_note='Documented synchronous/streaming output limit (thinking + output); '
                                           'background long decode is not used.'),
    KIMI_MODEL: dict(max_output_tokens=1048576, context_window=1048576,
                     source=KIMI_API_SOURCE, context_source=KIMI_MODEL_SOURCE, verified_at='2026-11-12',
                     capacity_note='Documented: max_completion_tokens defaults to 131072 and can be set up to '
                                   '1048576 (reasoning + answer); when input plus the limit exceeds the 1M '
                                   'context, the API lowers the limit for that request.'),
}


def validate_model(model):
    if model not in SUPPORTED_MODELS:
        raise ValueError('Unsupported experiment model: choose ' + ', '.join(SUPPORTED_MODELS))


def generation_settings(model):
    validate_model(model)
    settings = dict(**deepcopy(_API_CONTRACTS[model]),
                sampling_parameters={} if _API_CONTRACTS[model].get('provider_controls_sampling') else dict(temperature=0),
                stream=True, stream_options=dict(include_usage=True),
                transport_protocol=TRANSPORT_PROTOCOL, token_stall_seconds=TOKEN_STALL_SECONDS,
                request_timeout_seconds=API_REQUEST_TIMEOUT_SECONDS,
                complete_history_max_output=COMPLETE_HISTORY_MAX_OUTPUT,
                complete_history_thinking='enabled', reasoning_effort='provider_default',
                thinking_budget='provider_default',
                context_policy='Replay role/content only; do not replay reasoning_content')
    return settings


def provider_error_block(settings, status, message):
    """Name a contract-declared provider refusal delivered as an error (HTTP status and message), else None.

    Only declared forms are retried; every other provider error keeps its HTTP classification."""
    for rule in settings.get('blocked_errors') or ():
        if status in rule['http_status'] and isinstance(message, str) and any(m in message for m in rule['messages']):
            return rule['name']
    return None


def provider_block(settings, content):
    """Name a contract-declared provider safety refusal delivered as answer text, else None."""
    rule = settings.get('blocked_response')
    if rule and isinstance(content, str) and content.lstrip().startswith(rule['content_prefixes']):
        return 'blocked_content_text'
    return None


def agent_wire_messages(messages, settings, role, purpose):
    """Clarify only the JSON wire structure; leave caller-owned history intact."""
    if role not in ('task','meta') or purpose or not settings.get('agent_json_protocol'):
        return messages
    instruction=('JSON WIRE FORMAT: Return exactly one JSON object, without XML tags, Markdown or a list. '
        'For a tool call use exactly {"tool_name":"NAME","tool_input":{"PARAMETER":"VALUE"}}. '
        'tool_name must be a string and tool_input must be an object containing ALL arguments; '
        'never put command or any argument at the top level. Use only one available tool. '
        'Example: {"tool_name":"bash","tool_input":{"command":"pwd"}}. '
        'To finish without a tool use {"response":"your summary"}. '
        'This wire format supersedes <json> wrappers and XML examples; it does not change tool permissions or the task.')
    result=[dict(message) for message in messages]
    if result and result[0]['role']=='system':
        result[0]['content']=result[0]['content']+'\n\n'+instruction
    else:result.insert(0,dict(role='system',content=instruction))
    return result


def documented_limits(model):
    value = _DOCUMENTED_LIMITS.get(model)
    if value is None:
        return None
    return dict(model=model, **deepcopy(value),
                resolution='Official model specification; model-list metadata lacks capacity fields')


def load_credentials(path):
    path = Path(path)
    if path.suffix.lower() != '.csv':
        value = json.loads(path.read_text())
        if 'api_key_file' in value:
            if 'api_key' in value:
                raise ValueError('Choose api_key or api_key_file, not both')
            key_path = Path(value['api_key_file'])
            if not key_path.is_absolute():
                key_path = path.parent / key_path
            key = key_path.read_text().strip()
            if not key or any(c.isspace() for c in key):
                raise ValueError('Referenced API key must contain one nonempty token')
        else:
            key = value['api_key']
        return key, value.get('base_url') or value['api_base']
    rows = list(csv.reader(path.read_text(encoding='utf-8-sig').splitlines()))
    keys = [row[1].strip() for row in rows if len(row) == 2 and
            row[0].strip().lower() in ('apikey', 'api_key', 'api-key')]
    urls = [row[1].strip().rstrip('/') for row in rows if len(row) == 2 and
            row[1].strip().startswith('https://') and
            urlsplit(row[1].strip()).path.rstrip('/').endswith('/compatible-mode/v1')]
    if len(keys) != 1 or not keys[0] or len(urls) != 1:
        raise ValueError('CSV requires exactly one API key and one HTTPS OpenAI-compatible endpoint')
    return keys[0], urls[0]

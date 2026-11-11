"""Conservative sizing of the complete text-only validity request."""
from rsi.common import encoded


def judge_limits(metadata):
    """Use the provider maximum; Agent output settings do not cap the judge."""
    context, maximum = (metadata.get(key) for key in ('context_window', 'max_output_tokens'))
    if any(type(value) is not int or value < 1 for value in (context, maximum)):
        raise ValueError('Judge requires verified positive context_window and max_output_tokens')
    return dict(context_window=context, max_output_tokens=maximum, safety_tokens=1024)


def request_size(messages, limits):
    for key in ('context_window', 'max_output_tokens', 'safety_tokens'):
        if type(limits.get(key)) is not int or limits[key] < 1:
            raise ValueError('Missing verified judge limit: ' + key)
    # UTF-8 bytes conservatively bound ordinary byte-tokenized text, with
    # separate allowances for chat wrappers and provider overhead. This is
    # deliberately not reported as a model-specific exact token count.
    prompt = len(encoded(messages).encode()) + 1024
    total = prompt + limits['max_output_tokens'] + limits['safety_tokens']
    return dict(**limits, prompt_token_bound=prompt, total_token_bound=total,
                estimator='utf8_bytes_plus_1024_chat_allowance', fits=total <= limits['context_window'])


def require_context(messages, limits):
    size = request_size(messages, limits)
    if not size['fits']:
        raise ValueError(f"Judge request exceeds verified context: {size['total_token_bound']} > "
                         f"{size['context_window']} (conservative token bound)")
    return size

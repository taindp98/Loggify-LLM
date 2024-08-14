# Pricing per 1M tokens for different models
# Ref: https://openai.com/api/pricing/
unit_cost = {
    "gpt-3.5-turbo-instruct": {
        "prompt_tokens": 1.5,
        "completion_tokens": 2,
    },
    "gpt-3.5-turbo-0125": {
        "prompt_tokens": 0.5,
        "completion_tokens": 1.5,
    },
    "gpt-3.5-turbo-1106": {
        "prompt_tokens": 1.0,
        "completion_tokens": 2.0,
    },
    "gpt-4o-mini": {
        "prompt_tokens": 0.15,
        "completion_tokens": 0.6,
    },
    "gpt-4o-mini-2024-07-18": {
        "prompt_tokens": 0.15,
        "completion_tokens": 0.6,
    },
    "gpt-4-0125-preview": {
        "prompt_tokens": 10,
        "completion_tokens": 30,
    },
}

unit_cost_batch_api = {
    "gpt-3.5-turbo-0125": {
        "prompt_tokens": 0.25,
        "completion_tokens": 0.75,
    },
    "gpt-3.5-turbo-1106": {
        "prompt_tokens": 0.5,
        "completion_tokens": 1.0,
    },
    "gpt-4o-mini": {
        "prompt_tokens": 0.075,
        "completion_tokens": 0.3,
    },
    "gpt-4o-mini-2024-07-18": {
        "prompt_tokens": 0.075,
        "completion_tokens": 0.3,
    },
}

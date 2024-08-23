# Pricing per 1M tokens for different models
# Ref: https://openai.com/api/pricing/
unit_price = {
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
    "gpt-4o-2024-08-06": {
        "prompt_tokens": 2.5,
        "completion_tokens": 10,
    },
    "gpt-4-vision-preview": {
        "prompt_tokens": 10,
        "completion_tokens": 30,
    },
    "gpt-4o": {
        "prompt_tokens": 5,
        "completion_tokens": 15,
    }
}

supported_batch_api_llm_models = [
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
]

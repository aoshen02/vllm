# OpenAI-Compatible Server

vLLM provides an HTTP server that implements OpenAI's [Completions API](https://platform.openai.com/docs/api-reference/completions), [Chat API](https://platform.openai.com/docs/api-reference/chat), and more! This functionality lets you serve models and interact with them using an HTTP client.

!!! warning "API key authentication does not protect every endpoint"
    The `--api-key` option (or `VLLM_API_KEY` environment variable) only
    authenticates requests to endpoints under the `/v1`, `/v2`, and
    `/inference` path prefixes. Other endpoints on the same HTTP server are
    **not** authenticated — most notably `/invocations`, which exposes the same
    inference capabilities as the `/v1` endpoints. Do not rely on `--api-key`
    alone to secure vLLM. See
    [API Key Authentication Limitations](../../usage/security.md#api-key-authentication-limitations)
    for the full list of protected and unprotected endpoints and recommended
    hardening, such as deploying behind a reverse proxy.

## Supported APIs

We currently support the following OpenAI APIs:

- [Completions API](#completions-api) (`/v1/completions`)
    - Only applicable to [text generation models](../../models/generative_models.md).
    - *Note: `suffix` parameter is not supported.*
- [Chat Completions API](#chat-api) (`/v1/chat/completions`)
    - Only applicable to [text generation models](../../models/generative_models.md) with a [chat template](../online_serving/README.md#chat-template).
    - *Note: `user` parameter is ignored.*
    - *Note:* Setting the `parallel_tool_calls` parameter to `false` ensures vLLM only returns zero or one tool call per request. Setting it to `true` (the default) allows returning more than one tool call per request. There is no guarantee more than one tool call will be returned if this is set to `true`, as that behavior is model dependent and not all models are designed to support parallel tool calls.
- [Chat Completions batch API](#chat-api) (`/v1/chat/completions/batch`)
- [Responses API](#responses-api) (`/v1/responses`, `/v1/responses/{response_id}`, `/v1/responses/{response_id}/cancel`)
    - Only applicable to [text generation models](../../models/generative_models.md).
- [Embeddings API](../../models/pooling_models/embed.md#openai-compatible-embeddings-api) (`/v1/embeddings`)
    - Only applicable to [embedding models](../../models/pooling_models/embed.md).
- [Transcriptions API](./speech_to_text.md#transcriptions-api) (`/v1/audio/transcriptions`)
    - Only applicable to [Automatic Speech Recognition (ASR) models](../../models/supported_models.md#transcription).
- [Translation API](./speech_to_text.md#translations-api) (`/v1/audio/translations`)
    - Only applicable to [Automatic Speech Recognition (ASR) models](../../models/supported_models.md#transcription).

## Completions API

In your terminal, you can [install](../../getting_started/installation/README.md) vLLM, then start the server with the [`vllm serve`](../../configuration/serve_args.md) command. (You can also use our [Docker](../../deployment/docker.md) image.)

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123
```

To call the server, in your preferred text editor, create a script that uses an HTTP client. Include any messages that you want to send to the model. Then run that script. Below is an example script using the [official OpenAI Python client](https://github.com/openai/openai-python).

??? code

    ```python
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    completion = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "user", "content": "Hello!"},
        ],
    )

    print(completion.choices[0].message)
    ```

!!! tip
    vLLM supports some parameters that are not supported by OpenAI, `top_k` for example.
    You can pass these parameters to vLLM using the OpenAI client in the `extra_body` parameter of your requests, i.e. `extra_body={"top_k": 50}` for `top_k`.

!!! important
    By default, the server applies `generation_config.json` from the Hugging Face model repository if it exists. This means the default values of certain sampling parameters can be overridden by those recommended by the model creator.

    To disable this behavior, please pass `--generation-config vllm` when launching the server.

## Extra Parameters

vLLM supports a set of parameters that are not part of the OpenAI API.
In order to use them, you can pass them as extra parameters in the OpenAI client.
Or directly merge them into the JSON payload if you are using HTTP call directly.

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"},
    ],
    extra_body={
        "structured_outputs": {"choice": ["positive", "negative"]},
    },
)
```

## Extra HTTP Headers

The `X-Request-Id` HTTP request header can be enabled with
`--enable-request-id-headers`.

??? code

    ```python
    completion = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"},
        ],
        extra_headers={
            "x-request-id": "sentiment-classification-00001",
        },
    )
    print(completion._request_id)

    completion = client.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        prompt="A robot may not injure a human being",
        extra_headers={
            "x-request-id": "completion-test",
        },
    )
    print(completion._request_id)
    ```

The Completions, Chat Completions, and Responses APIs also support the
`X-Vllm-Priority` request header. Its value must be an integer and overrides
the `priority` value in the JSON request body. Non-zero priorities require the
server to use priority scheduling.

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_headers={"X-Vllm-Priority": "-10"},
)
```

## Returning token IDs (`return_token_ids`)

Set `return_token_ids: true` on `/v1/chat/completions` or `/v1/completions` to get back the exact prompt and output token IDs the engine used. RL and agentic rollouts can train on these IDs directly instead of retokenizing the text, which can drift from what the model saw (chat template, special tokens, non-unique tokenizations). No server flag is needed.

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"return_token_ids": True},
)
prompt_ids = completion.model_extra["prompt_token_ids"]
output_ids = completion.choices[0].model_extra["token_ids"]
```

Where the IDs appear:

| API | Prompt IDs (after chat templating) | Output IDs |
| --- | --- | --- |
| `/v1/chat/completions` | top-level `prompt_token_ids` | `choices[i].token_ids` |
| `/v1/completions` | `choices[i].prompt_token_ids` | `choices[i].token_ids` |

- **Streaming:** `prompt_token_ids` is sent only once: in the first chunk (Chat Completions) or the first chunk of each choice (Completions). Each chunk's `choices[i].token_ids` holds only the new tokens, so concatenate them. Chat Completions still emits a chunk when a token produces no text delta, so no IDs are lost.
- `token_ids` holds output tokens only, also with `echo: true`.
- Add `return_prompt_text: true` (Chat Completions) to also get the rendered prompt string as `prompt_text`.
- Each multi-turn request is re-templated from `messages`, so the `prompt_token_ids` of turn N+1 are not guaranteed to equal turn N's prompt plus output IDs. Use the returned IDs as the ground truth, or send token IDs directly with [`/inference/v1/generate`](token_in_token_out.md).
- **`include_reasoning: false` drops the IDs.** When the server has a reasoning or tool-call parser and the request sets `include_reasoning: false`, Chat Completions omits `choices[i].token_ids` and logprobs (streaming and non-streaming), so hidden reasoning tokens don't leak. `prompt_token_ids` is still returned. Keep the default `include_reasoning: true` for rollouts.
- With `return_weight_versions: true`, each choice also gets `weight_versions` spans over these output token indexes; see [Weight Versions in Outputs](../../training/async_rl.md#weight-versions-in-outputs).

!!! note
    `return_tokens_as_token_ids` (request field) and `--return-tokens-as-token-ids` (server flag) are different features. They only change the `token` strings inside `logprobs` to the form `token_id:<id>`, and only when logprobs are requested. They don't add `prompt_token_ids` or `token_ids`.

### Responses API

`/v1/responses` has no `return_token_ids` field. Set `enable_response_messages: true` instead; the response then carries `input_messages` and `output_messages`.

```python
response = client.responses.create(
    model="Qwen/Qwen3-8B",
    input="Hello!",
    extra_body={"enable_response_messages": True},
)
prompt_ids = response.model_extra["input_messages"][0]["tokens"]
output_ids = response.model_extra["output_messages"][0]["tokens"]
```

- For non-Harmony models, each entry is `{"type": "raw_message_tokens", "message": <text>, "tokens": [<ids>]}`. If `message` and `tokens` disagree, `tokens` is the source of truth. `input_messages[0].tokens` is the rendered prompt and `output_messages` holds the generated tokens, accumulated into one entry when streaming (they arrive on the final `response.completed` or `response.incomplete` event).
- For Harmony models (gpt-oss), `input_messages` and `output_messages` are Harmony message objects (role and content), not token ID lists. Use Chat Completions with `return_token_ids` if you need the raw IDs.
- `enable_response_messages` is documented as supported for non-background requests only (`background: false`, the default).
- `return_tokens_as_token_ids` has no per-request field here; only the server flag `--return-tokens-as-token-ids` applies to Responses logprobs (with `include: ["message.output_text.logprobs"]`).

## API Reference

### Completions API

Our Completions API is compatible with [OpenAI's Completions API](https://platform.openai.com/docs/api-reference/completions);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.

Code example: [examples/basic/online_serving/openai_completion_client.py](../../../examples/basic/online_serving/openai_completion_client.py)

#### Extra parameters

The following [sampling parameters](../../api/README.md#inference-parameters) are supported.

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/completion/protocol.py:completion-sampling-params"
    ```

The following extra parameters are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/completion/protocol.py:completion-extra-params"
    ```

### Chat API

Our Chat API is compatible with [OpenAI's Chat Completions API](https://platform.openai.com/docs/api-reference/chat);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.

We support both [Vision](https://platform.openai.com/docs/guides/vision)- and
[Audio](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in)-related parameters;
see our [Multimodal Inputs](../../features/multimodal_inputs.md) guide for more information.

- *Note: `image_url.detail` parameter is not supported.*

Code example: [examples/basic/online_serving/openai_chat_completion_client.py](../../../examples/basic/online_serving/openai_chat_completion_client.py)

#### Extra parameters

The following [sampling parameters](../../api/README.md#inference-parameters) are supported.

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/chat_completion/protocol.py:chat-completion-sampling-params"
    ```

The following extra parameters are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/chat_completion/protocol.py:chat-completion-extra-params"
    ```

### Responses API

Our Responses API is compatible with [OpenAI's Responses API](https://platform.openai.com/docs/api-reference/responses);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.

Code example: [examples/tool_calling/openai_responses_client_with_tools.py](../../../examples/tool_calling/openai_responses_client_with_tools.py)

To get prompt token IDs before generation, use
[`/v1/responses/render`](renderer.md#get-responses-prompt-token-ids). It preprocesses
a self-contained Responses request without inference and returns the prompt
token IDs, so a separate `/tokenize` call is not needed.

#### Sampling parameters

In addition to the OpenAI request fields, the Responses API forwards vLLM
[sampling parameters](../../api/README.md#inference-parameters) such as `top_k`,
`min_p`, `repetition_penalty`, `seed`, `stop`, and `ignore_eos` to the engine.

#### Extra parameters

The following extra parameters in the request object are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/responses/protocol.py:responses-extra-params"
    ```

The following extra parameters in the response object are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/responses/protocol.py:responses-response-extra-params"
    ```

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

### Logprobs for chosen token IDs (`logprob_token_ids`)

Set `logprob_token_ids` to get the logprob of a fixed set of token IDs that you choose (not the top-k) at every generated position.
At each position the response contains your IDs plus the sampled token, which is always included.

```python
label_ids = tokenizer.convert_tokens_to_ids(["yes", "no"])  # any in-vocab ids

# Chat Completions: `logprobs` must be true. The IDs replace `top_logprobs`, which is ignored.
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Is the sky blue? Answer yes or no."}],
    max_tokens=1,
    logprobs=True,
    extra_body={"logprob_token_ids": label_ids, "return_tokens_as_token_ids": True},
)
print(completion.choices[0].logprobs.content[0].top_logprobs)

# Completions: `logprobs` must be set (any integer); the IDs replace the top-k.
completion = client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="The sky is",
    max_tokens=4,
    logprobs=1,
    extra_body={"logprob_token_ids": label_ids},
)
```

Offline, pass the field to `SamplingParams`. `outputs[0].logprobs` then has one `{token_id: Logprob}` dict per generated position:

```python
params = SamplingParams(max_tokens=4, logprob_token_ids=label_ids)
```

Requirements and limits:

- At most 128 IDs (`MAX_LOGPROB_TOKEN_IDS`, independent of `--max-logprobs`). Each ID must satisfy `0 <= id < vocab_size`.
- Offline, leave `logprobs` unset or set it to `len(logprob_token_ids)`. Any other value raises a validation error, and a set value is also checked against `--max-logprobs`.
- `/v1/completions` rejects the request if `logprobs` is null. `/v1/chat/completions` rejects it unless `logprobs` is `true`.
- Not supported with `use_beam_search`. Completions also rejects `echo=True` with `max_tokens=0`, because no tokens are generated.
- With `return_tokens_as_token_ids: true`, entries appear as `token_id:<int>`, so you can map them back to your IDs. The values follow `--logprobs-mode` (default `raw_logprobs`).

!!! warning
    On the V1 model runner, if any request in a batch sets `logprob_token_ids`, the sampler replaces the logprobs of every request in that batch. Concurrent requests that asked for top-k `logprobs` lose their top-k entries; only their sampled token stays valid.
    Model Runner V2, which is the default, builds rows per request and does not have this problem.
    vLLM falls back to V1 when you set `VLLM_USE_V2_MODEL_RUNNER=0` or when a feature is unsupported; it logs "using the V1 model runner instead".
    If you must run on V1, serve `logprob_token_ids` traffic on a dedicated instance.

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

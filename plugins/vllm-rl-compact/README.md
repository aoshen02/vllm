# vllm-rl-compact

RL compact sample logprobs for `/inference/v1/generate` as a `vllm.endpoint_plugins`
plugin. Non-streaming requests that set `logprobs_format="compact"` get
`choices[].compact_logprobs`: the top-k of each position as base64 of little-endian
int32 ids and float32 logprobs (the sampled tokens are `choices[].token_ids`):

```json
{"num_positions": N, "num_slots": k, "dtype_token_ids": "int32",
 "dtype_logprobs": "float32", "byteorder": "little",
 "token_ids": "<base64 int32[N*k]>", "logprobs": "<base64 float32[N*k]>"}
```

`stream: true` and `logprobs: -1` with the compact format are 400. All other requests
are served by the unchanged core endpoint.

## Core APIs used

- the sample-logprobs container hook (`vllm.logprobs`:
  `register_sample_logprobs_container`, `set_sample_logprobs_container`,
  `SampleLogprobsHandle`);
- `ServingTokens.start_generate`;
- from `vllm/entrypoints/scale_out/token_in_token_out`:
  `logprobs_render.render_json_with_fragments`, `serving.build_response_off_loop`,
  `protocol.RenderedGenerateResponse` and `ArrayLogprobs._check_rows`.

No released vLLM has them yet, so `pyproject.toml` cannot state a lower bound.

## Enabling

- Install into the server's environment: `uv pip install ./plugins/vllm-rl-compact`.
  From a source tree: this directory on `PYTHONPATH`, plus the entry point written by
  `.venv/bin/python dev_entry_point.py DIR` and `DIR` on `PYTHONPATH` (the tests do
  this themselves).
- Allowlist: `VLLM_PLUGINS=rl_compact`. It is one allowlist for all plugin groups:
  name every other plugin that must still load.

The plugin's route takes the core route's place (`app.router.routes`), so
precedence, the single OpenAPI operation and `app.dependency_overrides` are kept;
requests without `logprobs_format="compact"` go to the core endpoint unchanged.

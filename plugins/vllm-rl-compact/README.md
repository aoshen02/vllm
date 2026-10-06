# vllm-rl-compact

RL compact sample logprobs for `/inference/v1/generate` as a `vllm.endpoint_plugins`
plugin. Requests that set `logprobs_format="compact"` get `choices[].compact_logprobs`:
the top-k of each position as base64 of little-endian int32 ids and float32 logprobs
(the sampled tokens are `choices[].token_ids`). There is one
compact format and no further switches. All other requests are served by the
unchanged core endpoint.

The compact format (DELTA slices of the core's `ArrayLogprobs` rows, wire encoder,
compact render, framing, request fields, route) is here; the row storage, the JSON
splice and the response-build threads are the core's (default-format array logprobs,
PR3). It needs these core APIs:

- the sample-logprobs container hook (`vllm.logprobs`:
  `register_sample_logprobs_container`, `set_sample_logprobs_container`,
  `SampleLogprobsHandle`);
- `ServingTokens.start_generate` (input processing and the engine call of the core
  handler, without its response build);
- `vllm.plugins.endpoint_plugins.routing.replace_route` (route replacement with
  conflict detection);
- from the default-format array logprobs (`vllm/entrypoints/scale_out/
  token_in_token_out`): `array_logprobs.ArrayLogprobs`,
  `logprobs_render.render_json_with_fragments`, `serving.build_response_off_loop`
  and `protocol.RenderedGenerateResponse`. The `ArrayLogprobs` subclass
  (`storage.py`) also uses the storage's protected members (`_append_rows`,
  `_check_rows`, `_filled_blocks`, the block fields), so it is tied to the core
  version it is developed against.

Other core names it uses are public by convention (no leading underscore):
`GenerateRequest` / `GenerateResponse` (subclassed or built), `GenerationError`,
`ErrorResponse`, `UsageInfo`, `PromptTokenUsageInfo`, `RequestResponseMetadata`,
`clamp_prompt_logprobs`, `should_include_usage`, `numpy2base64`, `as_list`, the
route decorators `with_cancellation` / `load_aware_call`,
the core handler's attributes `enable_prompt_tokens_details` / `enable_log_outputs` /
`request_logger` and its `create_error_response` / `create_streaming_error_response`,
and `app.state.serving_tokens`. It reproduces the core handler's three-line
`finish_reason == "error"` check rather than calling the protected `_raise_if_error`.

## Requirements

A vLLM build with the APIs listed above (the PR stack this plugin is developed
against: the sample-logprobs container hook, `ServingTokens.start_generate`,
`replace_route` and the default-format array logprobs). With an older vLLM the
plugin fails at import. No released vLLM version has them yet, so `pyproject.toml`
cannot state a lower bound.

## Enabling

- Install into the server's environment: `uv pip install ./plugins/vllm-rl-compact`.
  For development from the source tree without installing, put this directory on
  `PYTHONPATH` and make the entry point discoverable with
  `.venv/bin/python dev_entry_point.py DIR` plus `DIR` on `PYTHONPATH` (the tests do this
  themselves, see `tests/conftest.py`). Nothing generated is checked in.
- Allowlist: `VLLM_PLUGINS=rl_compact` (endpoint plugins never load without it).

**Caveat:** `VLLM_PLUGINS` is one allowlist for *all* plugin groups. With
`VLLM_PLUGINS=rl_compact`, every other installed plugin (general plugins, platform
plugins, ...) that is not named is no longer loaded. List them all
(`VLLM_PLUGINS=rl_compact,other_plugin`), or use a separate allowlist for endpoint
plugins once vLLM has one (proposal: `VLLM_ENDPOINT_PLUGINS` / `--endpoint-plugins`,
with `VLLM_PLUGINS` keeping its meaning for the other groups).

## Route

The plugin replaces the core `POST /inference/v1/generate` route with `replace_route`
(in place, so OpenAPI shows one operation with the plugin's request and response
schemas, and `app.dependency_overrides` apply). Non-compact requests are validated with this
plugin's request model (which adds `logprobs_format`) and then served by the replaced
core endpoint unchanged. The new route inherits the core route's dependencies. Startup
fails if the route cannot be replaced safely (another plugin already replaced it, a
catch-all or a `Host` route precedes it).

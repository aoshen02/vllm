# RFC：按位置查询指定 Token 的 Logprob

## 状态

草案（Draft）

## 摘要

增加一种仅用于推理/打分的接口：调用方可以为每个输入位置指定一组不同的
target token IDs，模型只返回这些 token 在对应位置的 logits 或 logprobs。

主要使用场景是 On-Policy Distillation（OPD）的 Top-K 打分：

1. Student rollout 在每个 response position 记录自己的 Top-K token IDs 和
   logprobs。
2. Teacher 在完整的 prompt 加 Student response 上执行一次 forward。
3. Teacher 在每个位置只 gather Student 提供的 candidate IDs，并返回对应的
   logprobs（或 logits）。

逻辑上的 candidate 张量形状为 `[batch, sequence_length, K]`。不同位置的
candidate 数量可以不同，因此实现需要支持 ragged 列表。

## 背景与动机

现有 `SamplingParams.logprob_token_ids` 支持每个 request 指定一组固定的
token IDs。这适合分类、label scoring 等场景：同一个 request 的所有 scored
rows 使用同一组候选。

OPD Top-K 的访问模式不同。一个 response 的候选集合可能是：

```text
position 0: [A, B, C]
position 1: [B, D, E]
position 2: [A, F, G]
```

如果把所有位置的 IDs 做 global union，再将 union 应用于每个位置，结果虽然
正确但会产生大量无用数据。设 response 长度为 `R`、每个位置的候选数为 `K`、
global union 大小为 `U`：

```text
实际需要的数据量：O(R × K)
global union 中间结果：O(R × U)
最坏情况：O(R² × K)
```

因此需要让每个位置只携带自己真正需要的 candidate IDs。

该 RFC 只定义通用的打分能力，不规定候选集合如何产生。候选可以来自：

- Student Top-K；
- Teacher Top-K；
- Student 与 Teacher Top-K 的 union/intersection；
- 调用方自定义的候选集合。

## 目标

- 在 prompt/prefill scoring 阶段，对每个位置的任意 token IDs 进行打分。
- 只返回调用方请求的 IDs 和对应数值，不产生完整 vocabulary 的 host/HTTP
  payload。
- 支持 raw logits 和 normalized logprobs，并遵循现有 `logprobs_mode` 语义。
- 保持现有 request-wise `logprob_token_ids` 行为不变。
- 支持 continuous batching 和不同位置候选数不同的 ragged 列表。
- 明确 causal 对齐：位置 `t` 的 candidate 对应“由位置 `t` 之前的 prefix
  预测位置 `t` 的 token”。

## 非目标

- 不在 vLLM 中实现 OPD 或任何蒸馏 loss。
- 不决定使用 Student Top-K 还是 Teacher Top-K。
- 不改变 token sampling，也不强制模型输出某个 token。
- 默认不返回完整 vocabulary logits。

## 建议接口

增加一个 engine-level 参数，暂定名称为 `logprob_token_ids_positions`：

```python
SamplingParams(
    max_tokens=0,
    logprob_token_ids_positions=[
        [],          # 不请求该位置
        [101, 202],  # 该位置的 candidate IDs
        [7, 8, 9],
    ],
)
```

具体 wire name 可由 API 维护者最终确定；关键语义是：每个 scored input
position 对应一组自己的 candidate IDs。对于 batch 中的多个 request，每个
request 都有自己的位置对齐列表。

返回结果应同时保留 IDs 和 values，并保持调用方传入的 candidate 顺序：

```text
position 0: ids=[],          values=[]
position 1: ids=[101, 202],  values=[...]
position 2: ids=[7, 8, 9],   values=[...]
```

第一个 causal position 可以保留空/占位项，以兼容现有 prompt logprob 输出
约定。调用方可以通过 `logprob_start_len` 限制返回区间。

## 执行模型

1. 模型对传入的 prompt 和 response token IDs 执行一次正常的 prompt/prefill
   forward。
2. Logits processor 根据配置执行必要的归一化。
3. GPU gather 读取：

   ```python
   logits[position, candidate_ids[position]]
   ```

   或对应的 logprobs。

4. 只将 ragged candidate 的 IDs 和 values 拷贝到输出。

对于 OPD Teacher 请求，prompt 位置使用空列表，response 每个位置携带
Student 的 candidate IDs。理想情况下 Teacher 不生成新 token；但当前 vLLM
调度器要求 `max_tokens >= 1`，且停止判断依赖至少一个 output token。因此
scoring-only 生命周期需要单独的请求模式；在该模式实现前，不能简单地把
`max_tokens=0` 当作现有可用接口。

注意：Teacher 仍然需要对完整的 prompt+response 做 forward。该接口只是避免
返回不需要的 token 分数，并不是只计算 response 的 hidden states。

## 兼容性与限制

- 现有 `logprob_token_ids` 继续作为“每个 request 一组固定 IDs”的快速路径。
- 初始实现建议只支持 prompt/prefill scoring；如果以后有明确需求，再扩展
  decode 路径。
- candidate IDs 必须进行 vocabulary 边界检查。
- 对重复 IDs 的行为需要明确：要么稳定保留重复项，要么直接拒绝。
- 输出必须保持调用方传入的 candidate 顺序。
- HTTP/OpenAI-compatible 暴露可后续单独讨论；嵌套列表会增加参数校验和
  序列化成本。

## 相关实现

Miles 的 OPD 实现有一条兼容旧 SGLang 的路径：先把所有位置的 candidate IDs
做 global union，再通过 request-wise `token_ids_logprob` 查询。例如各位置的
candidates 为 `[A, B]`、`[C, D]`、`[A, E]` 时，发送固定列表
`[A, B, C, D, E]`，Teacher 在每个位置都返回这五个 token 的 logprob，Miles
再按位置筛回真正需要的项。该路径无需服务端改动但会返回无用数据，规模为
`O(R × |U|)`（`U` 为 union 大小）。

可选的 `token_ids_logprob_positions` 路径则按位置发送 candidate 列表，Teacher
每个位置只 gather 自己的 IDs，将响应从 dense 的 `O(R × |U|)` 降为 sparse 的
`O(R × K)`。两条路径计算结果相同，区别仅在 candidate 的传输和 gather 范围。

SGLang 的 `top_logprobs_num` 已经可以返回每个位置自然 Top-K 的 IDs 和 values；
其旧版 `token_ids_logprob` 则是 request-wise 的固定 ID 列表。本 RFC 希望在
vLLM 中同时支持“按位置选择 candidate”和“按指定 ID gather”。

## 待讨论问题

- 参数名称使用 `logprob_token_ids_positions` 还是其他命名？
- raw logits 是否复用现有 `logprobs_mode`，还是引入独立的 scoring result？
- 是否先只提供 engine API，不暴露 OpenAI-compatible endpoint？
- 对 CUDA graph、TP/DP attention，ragged candidate 的最高效表示是什么？
- 位置对齐是否使用空列表，还是使用显式 start offset/packed `(position, IDs)`？

## 实施路线

### M1：Prefill request-wise fixed-ID scoring

先支持每个 request 一组固定 candidate IDs（global union）：

```python
prompt_logprob_token_ids=[101, 202, 303]
```

仅用于 scoring 请求。Teacher 对完整输入执行 forward，并在每个 scored
position gather 这组 IDs。该阶段复用现有的
request-wise token 状态和 GPU gather 路径，重点验证输出格式、logits/logprobs
语义以及 chunked prefill 下的位置对齐。

### M2：Prefill per-position sparse scoring

增加每个位置独立 candidate 列表的接口：

```python
prompt_logprob_token_ids_positions=[
    [], [101, 202], [7, 8, 9],
]
```

第一版仍限定为 prefill/scoring、非 streaming；暂不支持
decode 阶段的 per-position candidate、speculative/beam search、multimodal
输入、prefix cache 和 OpenAI-compatible endpoint。支持多 request batch、ragged
candidate 列表以及 chunked prefill。chunked prefill 必须携带每个 chunk 的绝对
起始位置，保证 logits 行与 candidate position 对齐；否则应显式拒绝请求，不能
静默返回错位结果。

内部可以先使用定宽的 `[num_rows, max_k]` candidate tensor 加 mask，输出时再
恢复为 ragged 列表，不要求第一版实现完全 packed 的 sparse tensor。

## M1 候选实现与实验基线

为避免过早绑定单一实现，M1 先保留三种可比较的原型：

1. **复用现有 prompt-logprob 输出**：在现有二维张量中保留实际 target 首列，
   后续列替换为 request 固定 IDs。改动较小，但必须避免把 custom candidate
   伪装成自然 Top-K 的 rank 语义。
2. **专用 scoring 输出**：新增只包含 `token_ids` 和 `logprobs` 的结果类型，
   与现有 `prompt_logprobs` 的 target/rank 约定隔离。语义最清晰，但需要额外
   接通 output 和异步拷贝链路。
3. **Full-vocab golden baseline**：使用现有 `prompt_logprobs=-1`，在调用方
   过滤所需 IDs。无需生产代码改动，用于 correctness 和通信/延迟对照，不能
   作为生产实现。

三种原型都只针对 v1 GPU runner 的 prefill/scoring；`max_tokens=0` 仍需单独的
scoring-only 请求生命周期，不能通过简单放宽参数校验实现。实验阶段如使用
`max_tokens=1`，必须明确丢弃生成 token，并将其视为临时 workaround。

初步 nightly 实验（`vllm/vllm-openai:nightly`，Qwen3-0.6B，单请求短输入）中，
full-vocab baseline 与 Hugging Face dense `log_softmax` 逐元素一致；仅选择 3 个
candidate 时仍产生约 4.6 MB dense payload，单次 warm run 约 0.89 s。因此它只适合
作为 correctness/golden baseline。基于语义审计，M1 主实现推荐专用 scoring 输出，
固定宽度 gather 仅作为小侵入对照原型。

# GLM 5.3 Flash API notes (M0)

Verified 2026-10-04 against the live API with a **Z.ai Coding Plan key**, model `glm-5.3-flash`, using `curl` (streaming and non-streaming). Raw stream bodies are in `fixtures/`. Request headers were never saved.

Spec section answered: `docs/superpowers/specs/2026-10-04-xolotl-v1-design.md` §16 (items 1-8). Evidence level is marked per row: **live** (observed on the wire), **docs** (Z.ai documentation only).

## Decisions

| # | Question (spec §16) | Decision | Evidence |
|---|---|---|---|
| 1 | Base URL / which endpoint the key belongs to | **Default `base_url` = `https://api.z.ai/api/coding/paas/v4`** (Coding Plan key). Endpoint is `{base_url}/chat/completions`, header `Authorization: Bearer <key>`. The spec's §13 default (`https://api.z.ai/api/paas/v4`) is wrong for this key; keep `base_url` configurable for pay-as-you-go keys. | live: Coding URL -> HTTP 200. Standard URL `https://api.z.ai/api/paas/v4` with the same key -> HTTP 429 `{"error":{"code":"1113","message":"Insufficient balance or no resource package. Please recharge."}}` (key is recognised, but has no balance on the pay-as-you-go API). docs: `docs.z.ai/devpack/tool/others` gives the Coding URL. |
| 2 | Resend `reasoning_content`? | **`RESEND_REASONING = true`**: resend each assistant turn's `reasoning_content` verbatim and unmodified, on turns with and without tool calls. It is not required for a 200, but without it the model loses its own thinking (see "Evidence for #2"). Cost: prompt grows by exactly the original `reasoning_tokens` of every resent turn. | live + docs, see below |
| 3 | Streamed tool-call delta format | **Both shapes exist; write one accumulator that handles both.** Default (no `tool_stream`): each call arrives **whole in one chunk**: `{"tool_calls":[{"id","index","type":"function","function":{"name","arguments":"<complete JSON string>"}}]}`. With the Z.ai-specific `"tool_stream": true`: **index-based partial `arguments`** (first chunk for an index carries `id`+`type`+`name`+first fragment; later chunks carry only `index`, `type`, `function.arguments` fragment). Accumulator: key by `index`; set `id`/`name` when present; append `arguments`. **Recommendation: do not send `tool_stream` in v1** (works without it; revisit if long `write` arguments make the UI feel stalled). | live: `fixtures/stream_tools.sse`, `fixtures/stream_two_tools.sse` (default) and `fixtures/stream_two_tools_partial.sse` (`tool_stream:true`). docs: `guides/capabilities/stream-tool`. |
| 4 | Usage field names | `usage` is a **top-level key on the final chunk** (the one that has `finish_reason`; it is followed by `data: [DONE]`). **Usage path:** `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`. **Cached-token path:** `usage.prompt_tokens_details.cached_tokens`. Also `usage.completion_tokens_details.reasoning_tokens` (a subset of `completion_tokens`). **No `stream_options` needed**; it is accepted (HTTP 200) but changes nothing. | live: `fixtures/stream_text.sse` last data line. docs: `usage.prompt_tokens_details.cached_tokens` "Number of tokens served from cache". |
| 4b | Cache behaviour | `cached_tokens` is counted in **64-token blocks** (observed 128, 192, 448, 7552, 7616) and is best-effort. Agent-loop-shaped growth (previous prompt + assistant + next user) hit: 7616 of 7634 prompt tokens cached. An identical ~7.6K prefix sent twice hit 7616/7618. Two requests that shared a 7.6K system prefix but differed only in the last user message got 0 cached on first try (3 of 3), yet a fresh system-prefix variant in the next run hit 7552, so hits are not predictable. Resent `reasoning_content` did not hurt cache hits. Treat the field as reliable only when non-zero. | live (probe g, g2, g3) |
| 5 | `reasoning_effort` values / default | **Valid (documented, accepted): `max`, `xhigh`, `high`, `medium`, `low`, `minimal`, `none`. Default: `max`.** The server does **not** reject unknown values: `"bogus"`, `""` and `"MAX"` all returned HTTP 200 and behaved like the default. So **validate client-side**. `none` does not turn thinking off (thinking cannot be disabled). Observed tiers (reasoning tokens, two runs each on one puzzle): none 19/24, minimal 17/26, low 23/24, medium 31/30, high 69/64, xhigh 58/85, max 73/61, default 65/111, bogus 63/70. | live (probe f, f2). docs: `api-reference/llm/chat-completion` lists the 7 values with default `max`; `guides/overview/concept-param` lists only `max`, `high`, `low` (older page; live API accepts all 7). |
| 6 | Context-overflow error | **HTTP 400, business code `1261`, message `Prompt too long`** (docs). Not reproduced live (see note below). **Recommended match:** `status == 400` AND (`error.code == "1261"` OR lowercased message contains `prompt too long` / `too long` / `max length` / `more than max tokens allowed`). Match the code first; the wording is the weaker signal. Do **not** match on bare `context` or `exceed` (too broad). | docs only: `api-reference/api-code` row `1261 / 400 / "Prompt too long"` (Chinese mirror `docs.bigmodel.cn/cn/faq/api-code`: 1261, 400, `Prompt 超长`). Third-party report of other wording (`tokens in request more than max tokens allowed`, GLM-5-Turbo on a Coding Plan, github.com/anomalyco/opencode/issues/27629), hence the fallback substrings. |
| 7 | Multiple tool calls per response / `parallel_tool_calls` | **Yes, multiple calls per response occur**: prompt "Call get_time for Tokyo and for Paris" returned two calls (`index` 0 and 1, distinct ids) in the default mode, in `tool_stream` mode, and in a non-streamed response. `parallel_tool_calls` is **not documented**; sending it returned HTTP 200 but the API also ignores any unknown field (`bogus_field_xyz` -> 200), so that is no evidence it exists. Do not send it. | live: `fixtures/stream_two_tools.sse`, `fixtures/stream_two_tools_partial.sse` |
| 8 | History with tool calls accepted without `tools`? | **Yes: HTTP 200.** The tools-disabled summary call (§10.4) can simply omit `tools`; no `tool_choice` workaround needed. (Also: `tools` + `"tool_choice":"none"` returned HTTP 200 although the docs say only `"auto"` is supported; effect not verified, so prefer omitting `tools`.) | live (probe e) |

### Extra decisions asked for

| Question | Answer | Evidence |
|---|---|---|
| Assistant message with `tool_calls`: is `content: ""` accepted? `null`? | **Both accepted (HTTP 200), and so is omitting `content`.** Keep sending `content: ""`. | live (probe d1/d3/d4) |
| Is `reasoning_content` on an assistant message accepted? | Yes, with or without it, with `tool_calls` or plain text; even an edited/truncated value is accepted (no integrity check, but see #2). | live (d1, d2, d7, d9) |
| Extra request field needed? | **None.** No `stream_options`. `tool_stream` is optional (see #3). Recommended sampling `temperature: 1.0`, `top_p: 0.95`, `reasoning_effort: "max"` is accepted. | live |
| Does `thinking` need setting? | No. Thinking is always on; `{"thinking":{"type":"disabled"}}` returns 200 but the model still reasons. `clear_thinking` made no observable difference to prompt tokens (see #2); not needed. | live (d5, d6, d10, f) |
| `max_tokens` limit | Range `[1, 131072]`; outside it -> HTTP 400 code `1210`. `finish_reason` is `"length"` when it truncates (reasoning counts toward it: `max_tokens: 30` produced 30 reasoning tokens and no content). | live |
| `finish_reason` values seen | `stop`, `tool_calls`, `length` | live |

## Evidence for #2 (`RESEND_REASONING`)

Docs (`guides/capabilities/thinking-mode`, via the fetch tool's extraction): preserved thinking is **enabled by default on the Coding Plan endpoint and disabled by default on the standard endpoint** (`clear_thinking: false` enables it); "you must return the complete, unmodified reasoning_content back to the API"; recommended for coding/agent scenarios; in tool loops thinking blocks should be "returned together with the tool results".

Live, same history (`user` -> `assistant{tool_calls}` -> `tool`), `prompt_tokens`:

| Variant | prompt_tokens | HTTP |
|---|---|---|
| no `reasoning_content` | 199 | 200 |
| with original `reasoning_content` (298 reasoning tokens in the source reply) | **497** (= 199 + 298) | 200 |
| with `reasoning_content`, `clear_thinking: true` | 497 | 200 |
| no `reasoning_content`, `clear_thinking: false` | 199 | 200 |
| tampered (half of it + " (edited)") | 353 | 200 |
| across a user turn: assistant text, no reasoning | 77 | 200 |
| across a user turn: assistant text + `reasoning_content` (171 tokens in source) | **248** (= 77 + 171) | 200 |

So the server ingests the resent reasoning (both inside a tool loop and across user turns), counts it exactly, and does not validate it.

Behavioural test (2 trials): the model was told to invent a 7-letter code word **only in its private thinking**, call `get_time`, and then report the word after the tool result. With the original reasoning resent it reported the word it had actually chosen (`QUORMEL`, `Zorblat`). With reasoning **not** resent it confabulated a different word both times (`Vexolyn`, `zumphex`). So without resend the model loses its own intermediate decisions.

Cache: B (no reasoning) and C (with reasoning) both cached 7616 tokens of ~7.64K, so resending does not break prefix caching, provided the text is byte-identical every turn (store it as received).

Cost: each resent turn adds its `reasoning_tokens` to the prompt (at the default `max` effort even "Say hi" produced 171). `estimate_tokens`/compaction must count it. A cheaper variant (resend only turns after the last user message) was not tested for quality; the docs ask for the full sequence.

## Stream format (from the fixtures)

- One `data: {json}` line per event, blank line between, **LF** line endings, terminated by `data: [DONE]`. No `event:` lines, comments or keep-alives were observed.
- Every chunk: `{"id","created","object":"chat.completion.chunk","model","choices":[{"index":0,"delta":{...}}]}`.
- Order: `delta.reasoning_content` chunks first, then `delta.content` chunks **or** one `delta.tool_calls` chunk per call (after the reasoning). Text/reasoning chunks carry `"role":"assistant"` on every chunk; the tool-call chunk has no `role`.
- Final chunk: `"finish_reason":"stop"|"tool_calls"|"length"`, `"delta":{"role":"assistant","content":""}`, plus `usage`.
- A response with a tool call has an empty `content` (`""`).
- Typical latency at default effort: 5-11 s for a short reply (reasoning dominates; 217-300 chunks for trivial prompts).

## Error format (live)

Envelope: `{"error":{"code":"<string>","message":"<text>"}}`; `code` is a **string**, the HTTP status carries the class.

| Case | HTTP | Body |
|---|---|---|
| Unknown model | 400 | `{"error":{"code":"1211","message":"Unknown Model, please check the model code."}}` |
| Empty `messages` | 400 | `{"error":{"code":"1214","message":"Input cannot be empty"}}` |
| `max_tokens` too large | 400 | `{"error":{"code":"1210","message":"The max_tokens parameter is illegal.：限制数值范围[1,131072]"}}` |
| Wrong key | 401 | `{"error":{"code":"401","message":"token expired or incorrect"}}` |
| No `Authorization` header | 401 | `{"error":{"code":"1001","message":"Authentication parameter not received in Header, unable to authenticate"}}` |
| No balance on pay-as-you-go API | 429 | `{"error":{"code":"1113","message":"Insufficient balance or no resource package. Please recharge."}}` |

**Retry caution (docs, not reproduced):** several non-transient quota errors also use HTTP 429: `1113` (no balance), `1308` (usage limit for the period), `1309` (Coding Plan expired), `1310` (weekly/monthly limit exhausted), `1311` (plan lacks the model), `1313`-`1321` (fair-use/enterprise limits). Only `1302` (rate limit) and `1305` (service overloaded) are transient. A blind "429 -> back off 5 times" would waste ~15 s on the permanent ones; classify by `error.code` and fail fast with the message for the rest.

## Note on item 6 (overflow was not sent live)

The docs state the code, status and message, so no oversized request was sent. The model's context window is 1M tokens (1,048,576 per Vercel's model page; the spec says 1M), so provoking a real overflow would cost over a million input tokens against the plan. The error envelope above was verified live with other 400s, which is why the recommended matcher keys on `error.code` first and keeps message substrings as a fallback.

## Fixtures (raw response bodies, byte-for-byte as received)

| File | Request | Notes |
|---|---|---|
| `fixtures/stream_text.sse` | `"Say hi"`, defaults | reasoning then text; usage 14/220, reasoning 171; 217 chunks |
| `fixtures/stream_tools.sse` | `"What time is it? Use the tool."`, one `get_time(city)` tool | reasoning then one whole tool call (`{"city":"London"}`), `finish_reason: tool_calls`; usage 177/310 |
| `fixtures/stream_two_tools.sse` | `"Call get_time for Tokyo and for Paris"` | two whole calls, `index` 0 and 1; usage 176/51, cached 128 |
| `fixtures/stream_two_tools_partial.sse` | same, with `"tool_stream": true` | extra fixture: index-based partial `arguments` (13 tool-call chunks) for the accumulator test |
| `fixtures/.gitattributes` | | `*.sse -text` so `core.autocrlf` cannot rewrite the bodies |

The `get_time` tool: `{"type":"function","function":{"name":"get_time","description":"Get the current time in a city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name, e.g. Tokyo"}},"required":["city"]}}}`.

## Sources relied on

- https://docs.z.ai/api-reference/llm/chat-completion (base URL, request fields, `reasoning_effort` values/default, `usage.prompt_tokens_details.cached_tokens`)
- https://docs.z.ai/api-reference/api-code (error-code table: 1261, 1113, 1302, 1305, 1308-1321)
- https://docs.bigmodel.cn/cn/faq/api-code (Chinese mirror of the error table)
- https://docs.z.ai/guides/capabilities/thinking-mode (preserved thinking, `clear_thinking`)
- https://docs.z.ai/guides/llm/glm-5.3-flash (1M context, 128K output, thinking cannot be disabled, `tool_stream` recommendation)
- https://docs.z.ai/guides/capabilities/stream-tool (`tool_stream`)
- https://docs.z.ai/devpack/tool/others (Coding Plan base URL)
- https://docs.z.ai/guides/overview/concept-param (older `reasoning_effort` list: max/high/low)
- https://vercel.com/ai-gateway/models/glm-5.3-flash/api (context window 1,048,576)
- https://github.com/anomalyco/opencode/issues/27629 (third-party overflow wording; no body captured)

Docs pages were read through a summarising fetch tool, so quoted doc text is as extracted, not copied from the raw page.

## Implications for the implementation tasks

- **Task 2 (`llm.rs`):** the existing parser already reads `usage.prompt_tokens`, `usage.completion_tokens`, `usage.prompt_tokens_details.cached_tokens` and `delta.reasoning_content` (paths are correct), and `apply_chunk` already keys tool calls by `index` and appends `arguments`, which covers both delta shapes (#3); the fixtures can drive its tests. Its provisional `OVERFLOW_MARKERS` (`context`, `too long`, `max length`, `exceed`, `1261`) is broader than #6 recommends (`context`/`exceed` are loose), and `classify` should split HTTP 429 by `error.code` (see "Retry caution"). No `stream_options` is needed (the current body omits it, correct).
- **Task 8 (`policy.rs`):** `RESEND_REASONING = true`.
- **Task 13 (config):** default `base_url = "https://api.z.ai/api/coding/paas/v4"`; validate `reasoning_effort` client-side against the 7 values.

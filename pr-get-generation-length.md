<!-- markdownlint-disable -->

## Disclosure

AI assistance was used. The changes were reviewed and tested manually.

## Overview

The harness reports scores, **but currently nothing about how much text the model produced to get them.** That makes two things invisible: how close a task is to `max_gen_toks` (a truncated run looks like a wrong answer), and **the cost of a reasoning model**, whose answer length says little about the length of the trace it wrote first.

This adds an opt-in `--generation-length` that reports, per task, the number of tokens in each generation: count, p25, mean, median, p75 and p90. It reads the untrimmed generation the backend already logs (`full_resps`, reasoning trace included) and tokenizes it with the tokenizer the API client was configured with.

## Claims

- `--generation-length` prints a `Generation length:` table after the results table, and stores the same numbers under `results["generation_length_stats"]`:

```
Generation length:
|Tasks         |n  |p25   |mean  |median|p75   |p90   |
|--------------|---|------|------|------|------|------|
|gsm8k_platinum|500|122.75|197.08|183.00|263.25|365.10|
|arc_easy      |N/A|N/A   |N/A   |N/A   |N/A   |N/A   |
```

carried into the results document — `<output_path>/<model>/results_<timestamp>.json`,
e.g. `results/genlen_qwen38_gsm8k_limit500/2026-10-26_17-57-12/Qwen__Qwen3.8-27B/results_2026-10-26T17-59-37.123456.json` — as:

```json
"generation_length_stats": {
  "gsm8k_platinum": {
    "count": 500,
    "mean": 197.078,
    "p25": 122.75,
    "median": 183.0,
    "p75": 263.25,
    "p90": 365.1
  },
  "arc_easy": {
    "count": 0,
    "mean": null,
    "p25": null,
    "median": null,
    "p75": null,
    "p90": null
  }
}
```

- Length is measured on `full_resps` when the backend logs it, i.e. on the generation before reasoning-trace stripping, and on the filtered response otherwise; both flat and `requests × repeats` sample shapes are accepted, and any other shape raises.
- Each task gets a row: tasks that do not generate (`loglikelihood` / multiple-choice) render as `N/A` with `count: 0` and null statistics, instead of being silently absent.
- A document is counted once even though it is logged once per filter, so `n` is the number of generated samples, not the number of filter records.
- The flag is an error, not a silent approximation, when it cannot measure anything, and the error type distinguishes the two cases: a backend that does not log the untrimmed generation raises `NotImplementedError` (the native `hf` / vLLM / SGLang clients today, to be supported in a follow-up), while a client without a local tokenizer or a run without `--log_samples` raises `ValueError`. The tokenizer message names the detected backend, because the default `tokenizer_backend=auto` selects the remote tokenizer and leaves `self.tokenizer` unset.
- Carries the small `CompletionText(str)` / `full_resp` plumbing (a `str` subclass, plus `full_resps` in the logged sample) that the measurement needs, for the OpenAI-compatible clients.

Left to other PRs:
- [ ] `full_resps` / `stop_reasons` / `found_answers` for the native backends (hf, vLLM, SGLang) is out of scope; those backends raise `NotImplementedError` (covered by a test), and the same `CompletionText` plumbing is proposed separately and would unblock them.
- [ ] No aggregation of generation length across task groups or across repeats (per-repeat lengths), and no inclusion of prompt length.

## Validation

### Unit tests

```
$ python -m pytest tests/test_generation_length.py -q
9 passed in 0.07s
```

All nine tests are new, in `tests/test_generation_length.py`:

| test | covers |
| --- | --- |
| `test_generation_length_prefers_full_resps` | `full_resps` wins over `resps`; flat and `requests × repeats` shapes both accepted |
| `test_generation_length_rejects_unexpected_shape` | `ValueError` on a sample whose responses are neither strings nor lists of strings |
| `test_generation_length_counts_each_document_once` | a document logged once per filter is counted once (`count == 1`) |
| `test_generation_length_is_na_for_non_generative_tasks` | `loglikelihood` task gets `count: 0` and null statistics |
| `test_generation_length_table_renders_na` | asserts the exact rendered table, including the `N/A` row |
| `test_percentile_interpolates` | p50 of `[1, 2, 3, 4]` is `2.5`; single value; p25 of `[0, 10]` is `2.5` |
| `test_generation_length_is_not_implemented_for_native_backends` | `NotImplementedError` for a backend that is not an OpenAI-compatible client, for both a bare object and a native-style backend that does have a tokenizer |
| `test_generation_length_requires_tokenizer` | `ValueError` naming the detected `tokenizer_backend` |
| `test_generation_length_requires_samples` | `ValueError` without `--log_samples` |

### End-to-end tests

8× MI350 (gfx950), ROCm/HIP 7.15, vLLM `0.32.1rc1.dev555+geca54bd1d` built from source, `Qwen/Qwen3.8-27B` (bf16, TP=8) served over `/v1/chat/completions`, `think_end_token=`, 5-shot `gsm8k_platinum`, `max_gen_toks=8192`, `num_concurrent=32`. Reproduce with:

```bash
# 1. serve the model
vllm serve Qwen/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B \
    --host 127.0.0.1 --port 8100 \
    --tensor-parallel-size 8 --dtype bfloat16 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --enable-prefix-caching > /tmp/qwen38_serve.log 2>&1 &

# 2. the measured run
export MODEL_DIR=/path/to/Qwen/Qwen3.8-27B
lm_eval --model local-chat-completions \
    --model_args "model=Qwen/Qwen3.8-27B,base_url=http://127.0.0.1:8100/v1/chat/completions,num_concurrent=32,timeout=99999,tokenized_requests=False,think_end_token=,tokenizer=$MODEL_DIR,tokenizer_backend=huggingface" \
    --tasks gsm8k_platinum --apply_chat_template \
    --gen_kwargs "max_gen_toks=8192" \
    --generation-length --log_samples \
    --output_path results/genlen_qwen38_gsm8k --limit 500

# 3. the N/A row: the chat client refuses loglikelihood, so use the completions client
lm_eval --model local-completions \
    --model_args "model=Qwen/Qwen3.8-27B,base_url=http://127.0.0.1:8100/v1/completions,num_concurrent=8,timeout=99999,tokenized_requests=False,tokenizer=$MODEL_DIR,tokenizer_backend=huggingface" \
    --tasks gsm8k_platinum,arc_easy \
    --gen_kwargs "max_gen_toks=2048" \
    --generation-length --log_samples \
    --output_path results/genlen_na_demo --limit 5
```

Result:

```
Generation length:
|Tasks         |n  |p25   |mean  |median|p75   |p90   |
|--------------|---|------|------|------|------|------|
|gsm8k_platinum|5  |118.00|147.00|122.00|145.00|218.20|
|arc_easy      |N/A|N/A   |N/A   |N/A   |N/A   |N/A   |
```

That run uses `local-completions`, since the chat client refuses `loglikelihood`; the chat client was used for the `gsm8k_platinum` run above.

## Details

- Percentiles use linear interpolation on the sorted lengths (`p25` of `[1, 2, 3, 4]` is `2.5`), so the reported p90 is not necessarily an observed length.
- `n` counts generations, so with `--repeats` it is `documents × repeats`; the per-repeat breakdown is not reported.
- Reviewers should look at the input validation rule and the overlap with the API-backend PR mentioned above: the `CompletionText` / `full_resps` plumbing is duplicated there deliberately so that this feature can land alone, and the duplicate disappears if that PR lands first.
- Rejected: measuring `resps` (the answer after trace stripping) — it undercounts exactly the models this exists for; rejected: deriving the length from `usage.completion_tokens` — the API reports it per request, not per generation, and not with repeats.

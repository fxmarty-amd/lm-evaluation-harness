<!-- markdownlint-disable -->

## Disclosure

AI assistance was used. The changes were reviewed and tested manually.

## Overview

The OpenAI-compatible backends (`local-completions`, `local-chat-completions`, `openai-completions`, `openai-chat-completions`) lag the native backends on reasoning-trace handling and on the metadata they log per sample.

Two behaviours were missing. `think_end_token` — already supported by `hf`, `vllm` and `sglang` — was not accepted, so a reasoning model could not have its `[THINK]…[/THINK]` trace stripped before answer extraction. And the task stop sequences were never passed down to the response parser, so `samples_*.jsonl` could not say whether a generation ended on a stop sequence, on EOS, or on the token cap.

## Claims

- All four OpenAI-compatible clients accept `think_end_token`. When it is set, task stop sequences are no longer sent to the server, because they can match inside the trace; they are applied to the answer instead.
- `stop_reasons`, `found_answers` and `full_resps` are now written for every generated sample by both the completions and the chat client. `found_answers` reports whether the thinking end marker was produced, `full_resps` keeps the untrimmed generation including the trace, and `resps`/`filtered_resps` keep the trimmed answer.
- `apply_chat_template` renders the tokenizer template whenever the tokenizer backend is `huggingface`, not only when `tokenized_requests` is set, so a configured chat template is no longer silently skipped.
- `local-chat-completions` sends its messages as JSON so the server applies the template itself, instead of sending a client-rendered string.

Left to other PRs:
- [ ] Diagnostics columns for missing/invalid answers are a separate PR.
- [ ] No end-to-end numbers here: this changes log metadata and prompt plumbing, not scores.

## Validation

```
$ python -m pytest tests/models/test_api.py -q
```

Coverage of the new metadata is currently through the end-to-end evals below rather than unit tests.

End-to-end, on 8× MI350 (ROCm 7.0), `gsm8k_platinum` with a chat-template checkpoint, 20 problems, greedy, reasoning enabled:

| client | API | stop_reasons present | trace in `full_resps` | answer in `resps` |
| --- | --- | --- | --- | --- |
| `local-chat-completions` | `/v1/chat/completions` | yes | yes | yes |
| `local-completions` | `/v1/completions` | yes | yes | yes |

Reproduce with the client switched by `--model local-chat-completions` / `local-completions`, `--model_args model=…,base_url=…/v1/chat/completions,think_end_token=[/THINK]`, then inspect `samples_*.jsonl`.

## Details

- `stop_reason` is derived in one place, `LocalCompletionsAPI._to_completion_text`, and used by both clients, so the two APIs cannot drift; `finish_reason == "length"` maps to `max_gen_toks`, a server-reported `stop_reason` wins over a suffix match, and an unmatched `stop` with an EOS configured is reported as `stop_sequence:<eos>`.
- The `max_tokens` sanity check added to the completions parser asserts a single choice per response, since OpenAI-compatible usage counts tokens across choices; servers returning `n > 1` will trip it.
- Reviewers should look at the `think_end_token` stop-sequence filtering: dropping task stops while the trace is enabled is what makes the answer-side post-processing correct, but it changes what is sent to the server.

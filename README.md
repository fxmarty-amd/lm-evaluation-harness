# lm-evaluation-harness, AMD fork

`lm-evaluation-harness` with the fixes we need for evaluating reasoning models served through an
OpenAI-compatible endpoint, up to date with upstream `main`. Each fix below is proposed upstream as
a separate PR; this branch is the combination, kept so our evals can use all of them at once.

| fix | upstream PR |
| --- | --- |
| `think_end_token`, stop reasons and sample metadata for the OpenAI-compatible backends (`local-completions`, `local-chat-completions`, `openai-*`) | https://github.com/EleutherAI/lm-evaluation-harness/pull/4473 |
| `answer-not-found` / `invalid-filter` diagnostic columns, with the offending samples dumped to `samples_not_found_*` / `samples_invalid_*` | https://github.com/EleutherAI/lm-evaluation-harness/pull/4475 (supersedes https://github.com/EleutherAI/lm-evaluation-harness/pull/3709) |
| repeat statistics (`--repeats N`, `_min/_mean/_median/_max_repeats`, per-repeat result directories) and `maj@k` for the AIME tasks | https://github.com/EleutherAI/lm-evaluation-harness/pull/4474 |
| `--generation-length`: token-length percentiles of the full generation (reasoning trace included) | https://github.com/EleutherAI/lm-evaluation-harness/pull/4477 |
| `gsm8k_platinum` description (the 5-shot prompt carried no instruction) and version bump | https://github.com/EleutherAI/lm-evaluation-harness/pull/3411 |
| `gpqa_diamond_generative_n_shot` prompt: expected final-answer format, few-shot answers rendered in that format, version bump | https://github.com/EleutherAI/lm-evaluation-harness/pull/3407 |
| `aime26` task (AIME 2026, `math-ai/aime26`), without the task-pinned stop sequences and greedy settings so `--gen_kwargs` can override them | https://github.com/EleutherAI/lm-evaluation-harness/pull/3603 |

Two behaviours worth knowing:

- the diagnostics, repeat reporting and `--generation-length` all rely on the per-generation metadata
  (`answer_found`, `stop_reasons`, `full_resps`) that only the OpenAI-compatible clients log, so they
  raise `NotImplementedError` on the native backends (`hf`, vLLM, SGLang) instead of reporting a
  misleading zero;
- `--repeats N` implies the per-repeat output: there is no separate flag.

## Quick example

```bash
lm_eval --model local-chat-completions \
    --model_args "model=<served-name>,base_url=http://127.0.0.1:8000/v1/chat/completions,tokenizer=<tokenizer-path>,tokenizer_backend=huggingface,think_end_token=</think>" \
    --tasks gsm8k_platinum --apply_chat_template \
    --gen_kwargs "max_gen_toks=8192" \
    --repeats 5 --generation-length --log_samples \
    --output_path results/gsm8k_repeats
```

which prints the results table with the two diagnostic columns, the `_repeats` variants, and a
`Generation length:` table (`n`, `p25`, `mean`, `median`, `p75`, `p90`), and writes each repeat under
`results/gsm8k_repeats/<task>_<timestamp>_repeat<N>/`.

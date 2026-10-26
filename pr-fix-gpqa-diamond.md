<!-- markdownlint-disable -->

## Disclosure

AI assistance was used. The changes were reviewed and tested manually.

## Overview

As per title: `gpqa_diamond_generative_n_shot` asks the model to answer with a letter and parses the answer with the `flexible-extract` / `strict-match` regexes, but its few-shot targets are bare letters (`B`). The demonstrations therefore do not show the answer format the filters expect, and no instruction tells the model to produce it.

This aligns the description and the few-shot targets with the expected format. Fixes #3404, see the context there. Same class of fix as the `gsm8k_platinum` description in #3411.

## Claims

- The task description now ends with the expected format: answers finish with `"The answer is (X)."` where X is the letter of the correct choice.
- The few-shot completions render as `The answer is {{answer}}.` via `fewshot_config.doc_to_target`, so the demonstrations match that instruction instead of being bare letters.
- `doc_to_target` for the scored question is unchanged, so the metric and filters behave exactly as before.
- `metadata.version` goes from `2.0` to `3.0`, so results produced with the previous prompt are not mixed with results after it.
- `lm_eval/tasks/gpqa/README.md` gains a version-history section recording what `3.0` changed, so the bump can be read without diffing the yaml.
- `strict-match` finally has something to match: 0.0000 → 0.5000 on `openai_gpt-oss-20b`, while `flexible-extract` stays flat (0.5455 → 0.5303).

Left to other PRs:
- [ ] No prompt/target change for the non-generative GPQA variants.
- [ ] `flexible-extract` moves slightly down in the measurement above (0.5455 → 0.5303, within ~0.4σ); this PR is about making the prompt consistent, not about a score gain.

## Validation

```bash
LOGLEVEL=debug CUDA_VISIBLE_DEVICES=0 lm_eval \
  --model hf \
  --model_args '{"pretrained":"/models/openai_gpt-oss-20b","dtype":"auto","chat_template_args":{"reasoning_effort":"low"},"enable_thinking": true,"think_end_token":200008}' \
  --device "cuda" \
  --gen_kwargs max_gen_toks=4048 \
  --tasks gpqa_diamond_generative_n_shot \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --log_samples \
  --output_path debug_gpqa \
  --num_fewshot 5 \
  --batch_size 16
```

on `main`:

```
|            Tasks             |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|------------------------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_diamond_generative_n_shot|      2|flexible-extract|     5|exact_match|↑  |0.5455|±  |0.0355|
|                              |       |strict-match    |     5|exact_match|↑  |0.0000|±  |0.0000|
```

with this fix:

```
|            Tasks             |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|------------------------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_diamond_generative_n_shot|      3|flexible-extract|     5|exact_match|↑  |0.5303|±  |0.0356|
|                              |       |strict-match    |     5|exact_match|↑  |0.5000|±  |0.0356|
```

(`Version` reads `3` here because of the version bump described in Claims; the two runs above were measured back to back on the same prompt apart from that metadata field.)

## Details

- Note: the `description` was modified too, not just the few-shot targets. Some models (`openai_gpt-oss-20b`) do not reliably obey an instruction that only says `following the format of the previous questions exactly`, even when the n-shot answers are in the right format, so the expected final-answer sentence is now stated explicitly.
- `fewshot_config.doc_to_target` is the mechanism the harness offers for rendering few-shot completions differently from the scored target, so the gold answer for scoring stays a single letter and needs no post-processing.
- Reviewers should check the wording once against the `strict-match` regex, since the filters accept both `(B)` and a trailing sentence.
- cc @baberabb

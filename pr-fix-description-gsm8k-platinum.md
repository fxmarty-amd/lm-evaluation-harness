<!-- markdownlint-disable -->

## Disclosure

AI assistance was used. The changes were reviewed and tested manually.

## Overview

`gsm8k_platinum` is a 5-shot task but defines no `description`, so the few-shot examples are presented with no instruction telling the model to answer the final question, or to follow the format of the example answers. The `description` is what the harness prepends as the system/first message when `--apply_chat_template` is used, so a chat model sees five worked examples and then a question with no statement of what to do with them.

This is the same problem that #2924 fixed for other tasks: fixes #2707. The wording follows the GPQA `n_shot` task, which already carries such a description: https://github.com/EleutherAI/lm-evaluation-harness/blob/fcddf195ec6bb69c63e36d54d75354f6ecaabab7/lm_eval/tasks/gpqa/n_shot/_gpqa_n_shot_yaml#L9

Adding the instruction, and bumping the task version so cached results for the old prompt are not silently reused.

## Claims

- `gsm8k_platinum` carries `description: "Here are some example math questions. Answer the final question yourself, following the format of the previous questions.\n"`, matching the wording of the other `gsm8k` variants and of the GPQA `n_shot` task.
- `metadata.version` goes from `3.0` to `4.0`, so results produced before this change are not mixed with results after it.
- The strict-match filter benefits most, since the model now produces the `#### <answer>` form the filter expects: on `openai_gpt-oss-20b` with `reasoning_effort=low`, `strict-match` goes from 0.0596 to 0.7072 while `flexible-extract` moves from 0.8983 to 0.9404.

Left to other PRs:
- [ ] The `until` / `max_gen_toks` of this task are untouched; changing them is a separate discussion.
- [ ] The scores below were measured with the pre-bump task version, on one model, at full `gsm8k_platinum`; they are evidence that the prompt was under-specified, not a benchmark of this PR across models.

## Validation

Before and after, `hf` backend, `openai_gpt-oss-20b`, `reasoning_effort=low`, `enable_thinking=True`, `think_end_token=200008`, `max_gen_toks=4048`, full `gsm8k_platinum`, `batch_size=16`.

On `main` (task version 3, no description):

```
|gsm8k_platinum|      3|flexible-extract|     5|exact_match|↑  |0.8983|±  |0.0087|
|              |       |strict-match    |     5|exact_match|↑  |0.0596|±  |0.0068|
```

With this change:

```
|gsm8k_platinum|      4|flexible-extract|     5|exact_match|↑  |0.9404|±  |0.0068|
|              |       |strict-match    |     5|exact_match|↑  |0.7072|±  |0.0131|
```

The task also loads and resolves:

```
$ python -c "from lm_eval.tasks import TaskManager; print(TaskManager().task_index['gsm8k_platinum'])"
```

## Details

- The version bump matters for anyone comparing against a cached run: the same task name now carries a different prompt, so old and new results must not be mixed.
- Reviewers should check the exact wording against the `strict-match` regex (`(?s).*#### `), since that filter is what the description is meant to make the model satisfy.
- cc @baberabb, cc @Qubitium (from the original description; both were involved in the prompt discussions).

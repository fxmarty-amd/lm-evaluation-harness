Branched of upstream d6de81643928d653435c431bae19945d41d32520.

Contains the following fixes:

## Summary of `answer-not-found`/`invalid-filter`

Displays:

```
|          Tasks          |Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|answer-not-found|invalid-filter|
|-------------------------|------:|----------------|-----:|-----------|---|----:|---|-----:|---------------:|-------------:|
|gpqa_diamond_cot_zeroshot|    2.3|flexible-extract|     0|exact_match|↑  | 0.84|±  |0.0524|          0.0800|        0.0000|
|                         |       |strict-match    |     0|exact_match|↑  | 0.82|±  |0.0549|          0.0800|        0.0000|
```

as well as details through `--log_samples --output_path ${DIRNAME}` logged json.

## Repeat support with min/max/median/mean

Displays:

```
|    Tasks     |Version|     Filter     |n-shot|          Metric          |   |Value |   |Stderr|answer-not-found|invalid-filter|
|--------------|------:|----------------|-----:|--------------------------|---|-----:|---|-----:|---------------:|-------------:|
|gsm8k_platinum|      3|flexible-extract|     5|exact_match               |↑  |0.9636|±  |0.0054|          0.0025|        0.0000|
|              |       |strict-match    |     5|exact_match               |↑  |0.9239|±  |0.0076|          0.0025|        0.0488|
|              |       |flexible-extract|     5|exact_match_max_repeats   |↑  |0.9686|   |      |          0.0025|        0.0000|
|              |       |strict-match    |     5|exact_match_max_repeats   |↑  |0.9239|   |      |          0.0025|        0.0488|
|              |       |flexible-extract|     5|exact_match_mean_repeats  |↑  |0.9649|   |      |          0.0025|        0.0000|
|              |       |strict-match    |     5|exact_match_mean_repeats  |↑  |0.9153|   |      |          0.0025|        0.0488|
|              |       |flexible-extract|     5|exact_match_median_repeats|↑  |0.9653|   |      |          0.0025|        0.0000|
|              |       |strict-match    |     5|exact_match_median_repeats|↑  |0.9107|   |      |          0.0025|        0.0488|
|              |       |flexible-extract|     5|exact_match_min_repeats   |↑  |0.9620|   |      |          0.0025|        0.0000|
|              |       |strict-match    |     5|exact_match_min_repeats   |↑  |0.9098|   |      |          0.0025|        0.0488|

Repeat 1/5
|    Tasks     |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|answer-not-found|invalid-filter|
|--------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|---------------:|-------------:|
|gsm8k_platinum|      3|flexible-extract|     5|exact_match|↑  |0.9636|±  |0.0054|          0.0017|        0.0000|
|              |       |strict-match    |     5|exact_match|↑  |0.9239|±  |0.0076|          0.0017|        0.0488|

Repeat 2/5
|    Tasks     |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|answer-not-found|invalid-filter|
|--------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|---------------:|-------------:|
|gsm8k_platinum|      3|flexible-extract|     5|exact_match|↑  |0.9653|±  |0.0053|          0.0008|        0.0000|
|              |       |strict-match    |     5|exact_match|↑  |0.9098|±  |0.0082|          0.0008|        0.0637|

Repeat 3/5
|    Tasks     |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|answer-not-found|invalid-filter|
|--------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|---------------:|-------------:|
|gsm8k_platinum|      3|flexible-extract|     5|exact_match|↑  |0.9686|±  |0.0050|          0.0008|        0.0000|
|              |       |strict-match    |     5|exact_match|↑  |0.9222|±  |0.0077|          0.0008|        0.0562|

Repeat 4/5
|    Tasks     |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|answer-not-found|invalid-filter|
|--------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|---------------:|-------------:|
|gsm8k_platinum|      3|flexible-extract|     5|exact_match|↑  |0.9620|±  |0.0055|          0.0000|        0.0000|
|              |       |strict-match    |     5|exact_match|↑  |0.9098|±  |0.0082|          0.0000|        0.0678|

Repeat 5/5
|    Tasks     |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|answer-not-found|invalid-filter|
|--------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|---------------:|-------------:|
|gsm8k_platinum|      3|flexible-extract|     5|exact_match|↑  |0.9653|±  |0.0053|          0.0000|        0.0000|
|              |       |strict-match    |     5|exact_match|↑  |0.9107|±  |0.0082|          0.0000|        0.0662|
```

with the repeat batched in the same eval run.

Log files through `--log_samples --output_path ${DIRNAME}` are separated for each repeat

## `think_end_token` support in `--model local-completions`

Self-explanatory

## Chat tempate fix in `local-completions`

Correctly apply chat tempate.

## Task definition fixes

Fix bugs in task definitions.

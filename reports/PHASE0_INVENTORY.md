# Phase 0 inventory and reconciliation

Nothing was deleted or overwritten. Every figure below is measured from the file named beside it.

## Reconciliations

### SFT_001 hallucination: 79.86% or 89.6%?

**MEASURED** - Both, on different evaluation splits. Neither figure is wrong; the split label was dropped when quoting them.

```
{
  "blind": {
    "n": 144,
    "accuracy_pct": 5.56,
    "hallucination_pct": 79.86
  },
  "dev": {
    "n": 200,
    "accuracy_pct": 5.0,
    "hallucination_pct": 89.58
  }
}
```

Source: `reports/SFT_EXPERIMENT_001.json`. Correction going forward: hallucination rate must always be quoted with its split and n.

### SFT_002 dataset size: 7,849 or 11,563?

**MEASURED** - Neither is the size of the dataset SFT_002 used. 7,849 is financial_sft_v1 (used by SFT_001). financial_sft_v2 holds 12159 rows, split 11563 train / 596 validation - 11,563 is the training split, not the dataset.

```
{
  "financial_sft_v1_rows": 7849,
  "financial_sft_v2_rows": 12159,
  "v2_train": 11563,
  "v2_val": 596
}
```

Source: `data/sft/*_manifest.json and reports/SFT_EXPERIMENT_002.json`. Correction going forward: quote dataset size and split separately.

## Datasets

| dataset | rows | sha256 |
| --- | --- | --- |
| `financial_sft_v1` | 7,849 | `329372998e3a...` |
| `financial_sft_v2` | 12,159 | `c50bf41e82ce...` |
| `tiny_overfit` | 54 | `1e538695a23e...` |
| `benchmark_dev` | 795 | `c00bbbe6c044...` |
| `benchmark_test_hidden` | 204 | `3aac98b056c3...` |
| `benchmark_blind` | 144 | `507d0956df44...` |
| `instruction_legacy` | 800 | `4c1e83d60f93...` |

## Checkpoints (all preserved)

| checkpoint | size | sha256 |
| --- | --- | --- |
| `baseline_247850` | 1,259 MB | `6e9e50ca59e5...` |
| `sft_001` | 445 MB | `4e76a3f8892f...` |
| `sft_002` | 445 MB | `d3d0419951b3...` |

## Chat conversion

`data\sft\financial_sft_chat.jsonl` - 12,159 rows, sha256 `6cc7e05c225c...`

Stored as role-based messages rather than a rendered prompt, so each base model's own chat template can render them. 42 replay rows carry raw text and no conversation, since they never had a question.

| task | rows |
| --- | --- |
| extraction | 5,826 |
| calculation | 3,000 |
| copy | 1,200 |
| abstention | 714 |
| interpretation | 443 |
| grounded | 438 |
| reasoning | 352 |
| definition | 124 |
| replay | 42 |
| explanation | 20 |

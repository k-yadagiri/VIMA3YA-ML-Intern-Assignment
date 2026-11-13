> ## ⚠️ DRAFT – NOT SUBMISSION-READY
> Sections marked ⟦PENDING⟧ need measured results that can only be produced by running `notebook.ipynb` on a Colab T4 (no GPU was available when this draft was written). Nothing below was invented: audit numbers are real (executed); every training/evaluation number is pending. Run notebook section 17 – it regenerates this file with measured values.

# NimbusPay Ticket Triage – LoRA/QLoRA Fine-Tuning Report
VIMA3YA · Machine Learning Intern take-home · all numbers below are **measured** (single run, seed 42, Colab T4) and read from `experiment_results.csv` / `logs/`.

## 1. Objective
Fine-tune an open ≤4B model with LoRA/QLoRA (≤50M trainable parameters, free T4) so that, from only the fixed system prompt and a ticket, it emits the 8-field triage JSON of `SCHEMA.md`. Inference: model's chat template, plain greedy `model.generate`, `max_new_tokens=200`, strip-only post-processing.

## 2. Base model
**Qwen/Qwen2.5-1.5B-Instruct** (≈1.5 (verified at runtime)B params): Apache-2.0 and ungated (reproducible for graders), native `system` role, digit-level tokenisation (good for 10-digit ids/amounts), small enough for five runs inside the 3-hour budget. Llama-3.2-1B was rejected (gated), Qwen2.5-3B (research licence, 2× runtime). SmolLM2-1.7B was *not* benchmarked – a limitation. Stack: HF Transformers + PEFT + bitsandbytes with a plain `Trainer` (own loss masking, unit-tested) because TRL's masking APIs vary by version and Qwen's template has no generation markers.

## 3. Data audit (Part A) and 4. Cleaning
Every one of the 1,445 train rows was parsed and re-checked. A SCHEMA-derived checker re-derives amount/date/id/channel/language from the ticket text and priority/needs_human from the schema formulas; discrepancies were classified, repaired **only if** the repaired record then agreed with the text on every field, else dropped. Result: **1,145 valid, 300 defective (175 fixed, 125 dropped) → 1,320 cleaned rows**; every changed row is in `cleaning_log.csv`.

| Defect (rows) | How found | Action |
|---|---|---|
| Truncated / invalid JSON (25) | `json.loads` fails | drop (reconstructing would invent a label) |
| Placeholder tickets "test", "asdf asdf", empty, yet fully-populated labels (10) | body is junk; label not derivable from text | drop |
| **Dev leakage**: train ticket text identical to a dev ticket (30) | exact text match (labels not read) | drop – training on dev is forbidden |
| Exact duplicate tickets (60 extra copies; no label conflicts) | text match | drop later copies |
| Key `prio` ∈ {low, medium, high} instead of `priority` (70) | key-set check; the words map 70/70 onto rule-derived P3/P2/P1 | fix: priority recomputed by rule |
| `priority` violates the schema rule (60: 31 P1→P3, 15 P3→P1, 14 P3→P2) | recompute from category/amount/legal threat; `needs_human` agrees with the *rule* priority in 100% of rows, so priority is the corrupted field | fix |
| `amount_paise` left in rupees / truncated (25) | label == text-derived paise // 100 in all 25 | fix |
| `txn_date` as DD/MM/YYYY (30); `txn_id` with spaces/hyphens/lower-case (20) | regex format checks | fix (normalise) |

Category labels were additionally screened with 10-fold cross-validated TF-IDF+LR (26 flagged, all read manually); all are schema-defensible, so none was changed (known boundary cases: PIN how-to `account_access`/`other`, tr-00190, tr-01377). Class balance: `payment_failed` 29%, `refund` 22%, `account_access` 13%, `kyc` 12%, `offers` 11%, `other` 7%, **`fraud` 6% (rarest, yet drives P1/needs_human)**; imbalance 4.7× before and after, proportions preserved. Hard-coded checker rules are used **only for cleaning**, never at inference.

## 5. Baselines (dev, 200 rows, original `score.py`)
⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧
Baseline 2 adds the full SCHEMA.md to the system prompt (allowed only here). ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧

## 6. Fine-tuning configuration
Prompt = original system+user via the chat template; loss only on the assistant JSON + end-of-turn token (prompt tokens = −100). Token analysis: lengths are bimodal – ≈3% of tickets (40 train/3 dev/11 test) are 7–9k-character quoted-thread emails – so `MAX_LEN=⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧` (longest sequence ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧ tokens), nothing truncated; memory is controlled with completion-only logits and length-grouped batches. LoRA r=16, α=32, dropout 0.05, all linear layers (≈18.5M trainable; every run asserts <50M), lr 2e-4 cosine, 5% warm-up, weight decay 0.01, micro-batch 4 × accumulation 4, 2 epochs, AdamW, fp16 AMP, gradient checkpointing, seed 42. QLoRA: NF4 + double quant, fp16 compute. Settings are reasoned defaults, not tuned on dev.

## 7. Four controlled experiments (one variable changed vs E0)
⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧
⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧
Dev has 200 rows (1 row = 0.5 pp; SE ≈ 3 pp), so differences below ~6 pp are **not** evidence. ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧

## 8. Best model
**⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧** – dev exact match **⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧**, mean field accuracy **⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧** (adapter on an fp16 base, plain greedy). Selected by a rule fixed beforehand: highest dev exact match, ties → mean field → lower memory. Adapter reloaded from disk onto a fresh base: ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧ ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧ Test predictions: 400/400 rows, ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧/400 JSON-valid (diagnostic only, nothing repaired).

## 9. Error analysis (best model, dev)
⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧

⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧

## 10. With one more week
(1) 3 seeds + paired bootstrap so deltas are credible; (2) rule-based **augmentation** (train+SCHEMA only) for rare surface forms: k/lakh/decimal amounts with distractors, year-less dates at year boundaries, spaced/hyphenated ids, legal-threat and prior-contact phrasings – the test set is "deliberately harder"; (3) adversarial validation train-vs-test on *inputs only* to see which features shift; (4) 3–4 epochs with a dev-loss stopping rule and a small LR sweep; (5) get a decision rule for schema-ambiguous boundaries (PIN how-to, double charge); (6) try Qwen2.5-3B with QLoRA if licensing allows.

## 11. AI tools used
⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧

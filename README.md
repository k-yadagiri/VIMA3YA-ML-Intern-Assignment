> ## ⚠️ DRAFT – NOT SUBMISSION-READY
> Sections marked ⟦PENDING⟧ need measured results that can only be produced by running `notebook.ipynb` on a Colab T4 (no GPU was available when this draft was written). Nothing below was invented: audit numbers are real (executed); every training/evaluation number is pending. Run notebook section 17 – it regenerates this file with measured values.

# VIMA3YA – NimbusPay Ticket Triage: LoRA Fine-Tuning of Qwen2.5-1.5B-Instruct

**Best result (one line):** `Qwen/Qwen2.5-1.5B-Instruct` + ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧ (⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧) → **dev exact match ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧, mean field accuracy ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧** on the 200 labelled dev rows with the official `score.py` (plain greedy `generate`, max_new_tokens=200, fp16 base).

## Business problem
NimbusPay (fictional payments app) needs support tickets triaged automatically into one JSON object with 8 fields (`category, priority, amount_paise, txn_date, txn_id, channel, language, needs_human`) following the business rules in the assignment's `SCHEMA.md`. At inference the model sees **only** the fixed system prompt and the ticket – the rules must be learned from the training data.

## Model
| | |
|---|---|
| Base model id | `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, ungated) |
| Architecture / size | decoder-only Transformer, 28 layers, hidden 1536, GQA (12 heads / 2 KV heads), ≈≈1.5 (verified at runtime)B parameters |
| Loaded in 4-bit? | **⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧** |
| Fine-tuning method | ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧; r=⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧, α=⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧, dropout 0.05, targets ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧; **⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧ trainable parameters** (<50M) |
| Libraries | transformers, peft, accelerate, bitsandbytes, PyTorch, scikit-learn (audit only) |
| Hardware | free Colab T4 (fp16 AMP; no bf16 on T4) |

## Dataset summary
Train 1,445 → audited and cleaned to 1,320 rows (175 fixed, 125 dropped; see `cleaning_log.csv` and `report.md`). Dev 200 clean labelled rows (measurement only; never trained on – 30 train rows that were verbatim copies of dev tickets were removed). Test 400 unlabeled rows → `predictions_test.jsonl`.

## How to run
1. Open `notebook.ipynb` in Google Colab, select **Runtime → T4 GPU**.
2. Upload `training_assessment.zip` to `/content` (or to `MyDrive/VIMA3YA-LLM-FineTuning/`).
3. **Runtime → Run all.** Results, adapters, logs and predictions are written to `MyDrive/VIMA3YA-LLM-FineTuning/`; finished runs are cached, so re-running after a disconnect resumes.
4. Download the executed notebook (File → Download .ipynb).

## Loading the adapter
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
base_id = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(base, "adapter").eval()
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TICKET}]      # original messages, nothing added
enc = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**enc, do_sample=False, max_new_tokens=200, pad_token_id=tok.pad_token_id)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```
Note: the model's own `generation_config` (Qwen2.5 ships a non-1.0 `repetition_penalty`) is used unchanged. Sensitivity check on 40 dev rows: ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧

## Project structure
```
VIMA3YA-LLM-FineTuning/
├── README.md
├── adapter/                 adapter_config.json, adapter_model.safetensors  (PEFT format)
├── notebook.ipynb           full pipeline with outputs
├── report.md                ≤3-page technical report
├── cleaning_log.csv         id,action,reason  (fix / drop)
├── predictions_test.jsonl   400 raw model outputs
├── experiment_results.csv   measured results of all runs
└── logs/                    training logs, score.py outputs, per-run results JSON
```

## Reproducibility
Seed 42 throughout; deterministic data audit; every experiment's config + measurements in `logs/`; test predictions generated at batch size 1 from the saved adapter; ⟦PENDING – filled automatically by notebook section 17 after the Colab run⟧ Raw assignment data is **not** included in this repository.

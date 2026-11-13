# VIMA3YA – NimbusPay Ticket Triage
## LLM Fine-Tuning Using QLoRA

**Project:** Machine Learning Intern – LLM Fine-Tuning Assignment  
**Base Model:** Qwen2.5-1.5B-Instruct  
**Task:** Automated customer support ticket classification and information extraction  
**Framework:** Hugging Face Transformers, PEFT, PyTorch

---

## 1. Executive Summary

This project focuses on developing an automated support-ticket triage system for NimbusPay, a fictional digital payments platform.

The objective is to fine-tune a lightweight open-weight Large Language Model (LLM) to classify customer support tickets and extract structured information into a standardized JSON format.

The solution uses Qwen2.5-1.5B-Instruct as its base model and a parameter-efficient fine-tuning approach using LoRA/QLoRA.

The final system is designed to generate eight structured fields:

- `category`
- `priority`
- `amount_paise`
- `txn_date`
- `txn_id`
- `channel`
- `language`
- `needs_human`

The implementation includes data auditing, preprocessing, baseline evaluation, parameter-efficient fine-tuning, controlled experiments, error analysis, and test-set prediction generation.

## 2. Business Problem

Customer support teams in digital payment platforms receive large volumes of tickets involving failed transactions, refunds, fraudulent activities, account access, KYC verification, and other payment-related issues.

Manual ticket classification can introduce delays and inconsistent prioritization.

The proposed solution aims to:

1. Automatically identify the ticket category.
2. Assign an appropriate priority level.
3. Extract transaction-related information.
4. Identify the customer's language.
5. Determine whether human intervention is required.
6. Generate standardized JSON responses.

## 3. Model Architecture and Configuration

| Parameter | Configuration |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Architecture | Decoder-only Transformer |
| Parameters | Approximately 1.5 billion |
| Fine-tuning approach | LoRA / QLoRA |
| Quantization | To be confirmed from execution |
| Maximum sequence length | As configured in notebook |
| Optimizer | As configured in notebook |
| Random seed | 42 |
| Training hardware | Google Colab T4 GPU |
| Framework | PyTorch and Hugging Face |

The project uses parameter-efficient fine-tuning to reduce GPU memory requirements while adapting the base model to the ticket-classification task.

## 4. Dataset Audit and Preprocessing

The dataset contains training, development, and test records.

| Dataset | Initial records | Purpose |
|---|---:|---|
| Training | 1,445 | Model development |
| Development | 200 | Evaluation |
| Test | 400 | Final prediction generation |

### Data audit results

The training dataset was audited for data-quality issues and inconsistencies.

| Audit operation | Records |
|---|---:|
| Original training records | 1,445 |
| Records fixed | 175 |
| Records dropped | 125 |
| Final cleaned training records | 1,320 |

The preprocessing workflow includes:

- Structural validation.
- Missing-value inspection.
- Label consistency checks.
- Duplicate and data-leakage inspection.
- Transaction identifier normalization.
- Date and amount validation.
- Cleaning-log generation.

The cleaning operations and their explanations are documented in `cleaning_log.csv`.

Development and test datasets are kept separate from model training.

## 5. Baseline Evaluation

Two baseline approaches are included in the evaluation workflow.

**Baseline 1 – Original Prompt**

The base model receives the original system prompt and customer ticket without additional business-rule instructions.

**Baseline 2 – Schema-Aware Prompt**

The base model receives the original prompt supplemented with the assignment's schema and business rules.

The evaluation uses:

- Exact Match Accuracy.
- Mean Field Accuracy.
- JSON validity.
- Field-level classification performance.

Actual evaluation measurements will be recorded from notebook execution.

## 6. Fine-Tuning Methodology

The proposed fine-tuning workflow uses Qwen2.5-1.5B-Instruct with parameter-efficient adaptation.

LoRA introduces trainable low-rank matrices into selected Transformer modules, reducing the number of parameters that need to be updated.

The training workflow includes:

1. Loading the pretrained model and tokenizer.
2. Preparing the cleaned training dataset.
3. Converting records into instruction-response examples.
4. Configuring LoRA/QLoRA.
5. Training using the configured optimization parameters.
6. Saving the trained adapter.
7. Evaluating the fine-tuned model.

The final inference prompt follows the assignment's prescribed format without including the schema rules or training examples.

## 7. Controlled Experiments

Four controlled experiments are included in the experimental design.

| Experiment | Comparison | Evaluation |
|---|---|---|
| Experiment 1 | Raw versus cleaned training data | Dev performance |
| Experiment 2 | LoRA versus QLoRA | Accuracy and GPU memory |
| Experiment 3 | Hyperparameter comparison | Dev performance |
| Experiment 4 | Custom experiment | Dev performance |

Each experiment is intended to record:

- Configuration.
- Training duration.
- GPU memory consumption.
- Development exact-match accuracy.
- Mean field accuracy.

The measured results will be documented in `experiment_results.csv`.

## 8. Evaluation and Error Analysis

The evaluation framework examines both overall prediction quality and individual field-level performance.

The analysis covers:

- Incorrect category predictions.
- Priority classification errors.
- Transaction amount extraction errors.
- Transaction date interpretation.
- Transaction ID normalization.
- Payment channel identification.
- Language classification.
- Human-escalation decisions.

At least ten actual prediction failures will be examined to identify recurring error patterns and potential improvements.

## 9. Test Prediction Generation

The final prediction workflow processes all 400 unlabeled test records using the selected fine-tuned model.

The required output format is JSONL:

```json
{"id":"ts-00001","output":"<raw model output>"}
```

Each test record must produce one corresponding prediction.

The final file will be named:

`predictions_test.jsonl`

Predictions must be generated using the assignment's prescribed inference configuration.

## 10. Reproducibility

The project is designed to be reproducible using Google Colab with a T4 GPU.

The notebook includes the workflow for:

- Dataset auditing.
- Data preprocessing.
- Baseline evaluation.
- Model fine-tuning.
- Controlled experiments.
- Evaluation.
- Error analysis.
- Test prediction generation.

The random seed is configured as 42.

## 11. Project Deliverables

The final GitHub repository is expected to contain:

```text
VIMA3YA-LLM-FineTuning/
│
├── README.md
├── notebook.ipynb
├── report.md
├── cleaning_log.csv
├── audit_lib.py
│
├── adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
│
├── predictions_test.jsonl
├── experiment_results.csv
│
└── logs/
    ├── audit_summary.json
    └── training and evaluation logs
```

Raw assignment datasets and full pretrained model weights are excluded from the repository.

## 12. Final Results

The final report will include the following measured results after successful execution:

| Metric | Result |
|---|---|
| Baseline 1 exact match | Pending execution |
| Baseline 2 exact match | Pending execution |
| Fine-tuned model exact match | Pending execution |
| Mean field accuracy | Pending execution |
| Trainable parameters | Pending execution |
| Training duration | Pending execution |
| GPU memory usage | Pending execution |
| Test predictions generated | Pending execution |

**Final status:** Data auditing and initial project preparation have been completed. Model training, experimental measurements, final adapter generation, and test prediction generation require execution and validation on the Colab T4 runtime.

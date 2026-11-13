# Outstanding items – this bundle is NOT submission-ready yet

No GPU and no Hugging Face Hub access were available while this bundle was prepared, so everything that requires them has **not** been produced (and nothing was faked):

| Required deliverable | Status | How to get it |
|---|---|---|
| `adapter/adapter_config.json`, `adapter/adapter_model.safetensors` | **MISSING** | Colab T4 → run `notebook.ipynb`; section 14 writes `final/adapter/` to Drive |
| `predictions_test.jsonl` (400 rows) | **MISSING** | notebook section 16 |
| Baseline 1/2 scores, 5 training runs, experiment table, `experiment_results.csv` | **NOT RUN** | notebook sections 11–13 |
| Error analysis with ≥10 real dev failures | **NOT RUN** | notebook section 15 |
| `README.md`, `report.md` final numbers | **DRAFT (⟦PENDING⟧ slots)** | notebook section 17 regenerates both from measured results |
| `notebook.ipynb` with GPU outputs | **PARTIAL** – Part A (audit, cleaning, log) executed with real outputs; GPU cells unexecuted | after the Colab run: File → Download → .ipynb |

Done and verified here: data audit, cleaning, `cleaning_log.csv` (300 rows: 175 fix, 125 drop), `logs/audit_summary.json`.

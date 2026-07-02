# Utilities

`utils/` contains shared helpers used across collection, training,
certification, runtime selection, and reporting.

## Key Modules

| File | Purpose |
|---|---|
| `cone_utils.py` | Simplex and worst-case motive helpers |
| `return_targets.py` | Discounted and doubly-robust return target helpers |P
| `support_geometry.py` | Support-function utilities and basis directions |
| `weight_set_store.py` | Context-conditioned `W_x` vertex/support storage |
| `mdn_selection.py` | Alpha-to-weight conversion and candidate scoring |
| `mdn_contracts.py` | MDN decision/candidate record contracts |
| `mdn_record_builder.py` | Build MDN training and evaluation records |
| `mdn_data_adapter.py` | Convert rollout/candidate-set files into MDN records |
| `mdn_checkpoint_loader.py` | Shape-inferred MDN checkpoint loading |
| `mdn_stub.py` | Deterministic MDN fallback for tests and smoke runs |
| `mdn_runtime_pipeline.py` | Runtime certification with `W_x` tracking |
| `admission_report.py` | JSON/Markdown admission report generation |

## Tests

```bash
python -m pytest tests/test_mdn_contracts.py tests/test_mdn_data_adapter.py -v
python -m pytest tests/test_support_geometry.py tests/test_weight_set_store.py -v
python -m pytest tests/test_admission_report.py -v
```

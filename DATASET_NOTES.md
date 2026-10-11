# Dataset Notes — IntelliQ

## Record Counts
| Category | Count |
|----------|-------|
| Total rows in `queue_data.csv` | 1,552 |
| Original records (real observations, integer wait_time) | 586 |
| Simulated records (synthetic augmentation, decimal wait_time) | 966 |
| Training set (original 80% + all simulated) | 1,435 |
| Test set (original 20% only) | 117 |

## Column Definitions
- `arrival_time` — customer arrival timestamp (`DD-MM-YYYY HH.MM` format)
- `start_time` — service start timestamp (same format)
- `finish_time` — service end timestamp (`YYYY-MM-DD HH:MM:SS` format, some missing)
- `wait_time` — **`start_time − arrival_time` in minutes** (prediction target)
- `queue_length` — number of customers in queue at arrival

## Target Definition
**`wait_time = start_time − arrival_time` (in minutes)**

This is consistently applied across all records. The original column values have been
recalculated using this formula. `finish_time − start_time` is service duration — it is
**NOT** the waiting time and is not used as a feature (avoiding target leakage).

## Data Quality
- No invalid timestamps found
- 8 rows have missing `finish_time` (retained, as finish_time is not a feature)
- 2 duplicate rows detected
- 0 rows with negative wait time
- 0 rows dropped after cleaning

## Timestamp Format
The dataset uses `DD-MM-YYYY HH.MM` format where `.MM` represents minutes (not
decimal fractions). For example, `30-03-2023 00.10` means 00:10 (midnight and 10 minutes),
NOT 00:06 (midnight and 6/100 of an hour).

## Simulated Records
Simulated rows are **synthetic examples** generated from observed statistical patterns
(queue length, hour-of-day, wait time distributions). They are:
- Added to the **training set only**
- **Not** used in the test set
- Labeled by decimal wait_time values
- Clearly distinct from real observations

**Do not present simulated-data evaluation as verified real-world accuracy.**

## Files
- `queue_data.csv` — cleaned dataset (5 columns, no hash or source identifier columns)
- `queue_data_original.csv` — untouched backup of the original CSV
- `queue_data_backup.csv` — additional backup copy
- `model_comparison.csv` — MAE, RMSE, R² for both models
- `test_predictions.csv` — predictions on original test records
- `training_summary.json` — full pipeline summary with cleaning report

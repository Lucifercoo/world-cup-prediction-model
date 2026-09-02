# Base Model vs Realtime-Assisted System

Both rows use the same 79 pre-kickoff forecasts and regulation-time results.
The realtime-assisted system used human-collected evidence interpreted with GPT-5.5
at the `very high` reasoning setting.

| System | Matches | Outcome | Top1 total | Top2 total | Any exact | Any score bucket | Median deviation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base model | 79 | 68.4% | 21.5% | 57.0% | 25.3% | 57.0% | 0.667 |
| Realtime-assisted | 79 | 67.1% | 32.9% | 69.6% | 35.4% | 77.2% | 0.700 |

Realtime team context was applied to 75/79 matches; pre-match shape context to 72/79.
Different search dates, sources, or language models can produce different realtime inputs.

## Score Columns

The base model has three historical score columns; the realtime-assisted system has four.

| System | Score | Available | Exact | Outcome | Bucket | Median deviation |
|---|---|---:|---:|---:|---:|---:|
| Base model | model | 79 | 7.6% | 68.4% | 21.5% | 0.667 |
| Base model | aggressive | 79 | 13.9% | 68.4% | 35.4% | 0.750 |
| Base model | market | 79 | 6.3% | 70.9% | 15.2% | 1.000 |
| Realtime-assisted | model | 79 | 13.9% | 67.1% | 32.9% | 0.667 |
| Realtime-assisted | aggressive | 79 | 13.9% | 67.1% | 36.7% | 0.600 |
| Realtime-assisted | market | 79 | 8.9% | 59.5% | 30.4% | 0.667 |
| Realtime-assisted | upset | 71 | 5.6% | 21.1% | 42.3% | 1.000 |

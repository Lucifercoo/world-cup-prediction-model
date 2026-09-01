# Experiment Registry

Files in `experiments/` are isolated diagnostics. They may read formal model
outputs, but they write only experiment reports and never update production
parameters or source data.

List the registered experiments:

```powershell
uv run python -m wc_model experiment list
```

Run one experiment by its stable name:

```powershell
uv run python -m wc_model experiment run <name>
```

## Experiments

| Name | Purpose | Required local inputs |
|---|---|---|
| `early-knockout-total-cap` | Test high-goal caps in R32 and R16 | Finished cache evaluation CSV |
| `joint-total-score` | Evaluate total-goal and score coverage jointly | Current realtime plan and results |
| `knockout-draw-score` | Test draw-oriented knockout score allocation | Realtime cache and finished evaluation |
| `knockout-strong-favorite-total` | Test knockout controls for strong favorites | Realtime cache and finished evaluation |
| `low-block-effect` | Measure aggregate effects of low-block labels | Current realtime plan and results |
| `low-block-mechanism` | Compare low-block adjustment mechanisms | Full private model inputs |
| `outcome-conflict-score` | Test score allocation under outcome conflicts | Current realtime plan and results |
| `strong-favorite-low-bucket-grid` | Search strong-favorite protection thresholds | Full private model inputs |
| `total-goal-bucket-score` | Compare score selection inside total-goal buckets | Historical and current prediction inputs |
| `xg-model-legacy-backup` | Compare xG model scores with the legacy backup | Current realtime plan and results |
| `xg-score-generation` | Evaluate xG score-generation variants | Current realtime plan and results |

## Interpretation

An experiment result is evidence, not configuration. A proposed model change
must still pass the strict pre-kickoff evaluation rules in `docs/DESIGN.md` and
show the affected matches and aggregate metrics before its parameters are
changed. Missing inputs must fail explicitly; experiments do not generate mock
or fallback data.

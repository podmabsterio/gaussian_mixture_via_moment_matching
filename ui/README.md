# GMM Experiment Studio

Schema-driven web UI for the existing experiment pipeline in the repository root.
The scientific code is not duplicated: every submitted form is compiled into the
same Hydra/OmegaConf structure consumed by `run.py` and executed in an isolated
worker process.

## Start

From the repository root:

```bash
python -m ui.start
```

Open <http://127.0.0.1:8080>.

## Declarations

- `declarations/models_declaration.json` contains the currently supported model,
  `src_np.gmm.MomentGaussianMixtureModel`.
- `declarations/datasets_declaration.json` contains dataset generators and their
  editable parameters.
- `declarations/execution_parameters.json` contains runner/output parameters plus
  quick-run parameters, metric declarations, and aggregation configuration.

Every parameter has its own UI default. It does not have to match the Python
constructor default. Optional metadata:

- `advanced: true` hides the control until the global **Advanced** switch is on.
- `hidden: true` prevents the control from ever reaching the frontend form; the
  backend always injects the declaration's `default` and rejects client attempts
  to override it.

Supported field types are `integer`, `number`, `boolean`, `string`, `enum`,
`integer_array`, and `number_array`. Numeric limits are enforced in both the
browser and backend.

## API

- `GET /api/declarations`
- `POST /api/data-preview`
- `GET /api/comparison`
- `POST /api/runs`
- `GET /api/runs`
- `GET /api/runs/<id>`
- `GET /api/runs/<id>/results`
- `POST /api/runs/<id>/cancel`
- `POST /api/quick-runs`
- `GET /api/quick-runs/<id>`
- `GET /api/quick-runs/<id>/events`
- `POST /api/quick-runs/<id>/cancel`
- `POST /api/quick-runs/<id>/save`
- `GET /api/saved-runs`
- `GET /api/saved-runs/<run_name>`

The data-preview endpoint instantiates the selected declared generator, creates
the complete sample, and computes PCA plus projected component covariances on
the backend. It returns at most 3,000 display points together with component
centers and one-standard-deviation ellipse geometry; the browser only renders
that prepared result.

The comparison endpoint reads paired aggregate files from every saved dataset:
`mean.csv` with `std.csv`, or `median.csv` with `iqr.csv`. It filters by optional
run, dataset, and model names and averages available central values and spreads
by model, dataset, or run name. Missing metrics and spreads are omitted rather
than replaced with zero. Since the persisted IQR is a width (`Q3 - Q1`) rather
than separate quartiles, its chart whisker uses that total width centered on the
median. Comparison is intentionally independent of the in-memory list of jobs
started by the current UI server process.

Named persistent runs are saved through the project's existing `init_saving()`
implementation under `results/<run_name>`. Unnamed runs receive a generated name.
When **Keep result** is disabled, output is placed under ignored `.ui_runtime/`.

Clicking a run in the **Results** tab opens a read-only detail view. Its
Configuration section reuses the declaration-driven Experiment controls with
the compiled values returned by the run API; the global **Advanced** switch
continues to control advanced fields. Aggregate metrics, the worker log, and a
collapsible raw compiled configuration are available alongside that view.
The list merges active in-memory jobs with every valid
`results/<run_name>/config.yaml` found on disk and de-duplicates persistent
runs that are also present in the current server session.

## Quick experiments

The **Quick experiment** tab runs one generated dataset and one model seed. Its
loss, JSON-declared metrics, and estimated component geometry are reported after
each outer optimizer iteration. The backend fits PCA once to the generated data
and uses that same basis for every true and estimated center/covariance snapshot.
The browser receives already projected points and one-standard-deviation ellipse
geometry through a reconnectable server-sent-event stream.

Quick runs are temporary until **Save to results** is clicked. Saving creates the
same `results/<run_name>/<dataset>/{mean,std,median,iqr,raw}.csv` layout consumed
by Results and Compare, plus `quick_trace.json` with the iteration timeline.

Iterative mixture models expose progress through an optional callback; no UI
base class is required. The shared types are exported by `src_np`:

```python
from src_np import IterationSnapshot, MixtureParameters

def fit(self, X, *, iteration_callback=None):
    # ... update one meaningful outer iteration ...
    if iteration_callback is not None:
        iteration_callback(
            IterationSnapshot(
                iteration=iteration,
                loss=loss,
                parameters=MixtureParameters(means, weights, covariances),
            )
        )
```

Callbacks are optional and ordinary batch execution does not install one. A
snapshot must own copies of canonical `means`, `weights`, and full
`covariances`; the provided dataclasses validate and defensively freeze them.

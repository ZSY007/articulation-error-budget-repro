# Anonymous reproduction package

This is the experiment release accompanying *Articulation Error Budgets Depend on the
Execution Criterion and Reference Update: A Simulation Study*. It contains object IDs,
stable seeds/configuration manifests, saved out-of-fold predictions, execution outcomes,
reset events, all E3 per-step traces, analysis scripts, and the common-support correction.

For saved-result verification, create a Python 3.10 environment, install
    `requirements-analysis.txt`, and run:

    python code/analysis/verify_release.py
    python code/analysis/reproduce_saved_results.py

The command writes `verification/reproduced_headlines.json`. It does not refit models or
run physics. The original analysis and experiment drivers are under `code/original/`.
Exact folds and fixed OOF predictions are in `data/p1/`; the rendered-RGB saved predictions
and correction tables are in `data/p2/`; paired delay trials, reset events, and compressed
per-step traces are in `data/p3/`. See `DATA_ACCESS.md` before attempting physics reruns.

Scope: all manipulation executions are simulated. The real-video material is a diagnostic
failure case and contains no robot operation.

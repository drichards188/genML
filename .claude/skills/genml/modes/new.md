# New competition (interactive)

Set up a Kaggle tabular competition end to end, so the autopilot can start on it. Arguments: the
competition slug (for example `playground-series-s6e12`) and optionally a short problem name.

1. Create the problem: `.venv/bin/python src/genML/main.py --init-kaggle <slug> --name <name>`. Read the
   written `problems/<name>/problem.yaml` against the competition's evaluation page: target, id column,
   task, metric (declare it), positive class. Choose `cv` to match how the model would be used: stratified
   5-fold for independent rows, a time-based split when rows are ordered in time, a group split when one
   customer or entity has many rows (`cv.method`: `stratified_kfold` or `kfold`, `group_kfold` / `stratified_group_kfold` with
   `cv.group`, or `time_blocks` with `cv.time_column`). Start with `feature_engineering.base: raw` and `models: [XGBoost, LightGBM, CatBoost]`.
2. Download the data: `.venv/bin/python src/genML/main.py --problem <name> --download` is the full run; to only
   fetch data, use kagglehub as `pipeline/problem.py: locate_data_dir` expects. Never copy Kaggle data into the repo.
3. Research, then write `problems/<name>/notes.md` with facts only, each with its source and date:
   deadline and submission limit; rows and columns; whether an original dataset exists (and its size);
   the leaderboard spread (#1, #50, median), and how the test data relates to train (later in time, other
   customers). Do not read solutions, write-ups, discussions or public notebooks for ideas: features must come
   from domain knowledge or genML's own analysis (see CLAUDE.md, purpose).
4. Baseline: `--baseline <name>` queues one run of the spec's settings followed by a seed test (minutes: the run's best model refit with two more seeds). Its spread is the noise the judge needs before the
   autopilot can start. Check the spec declares
   `feature_engineering.base: raw` first: the judge refuses a baseline that selects features with labels.
5. Offer to start the autopilot (`/genml autopilot <name>`), with a directive if the notes point to an
   obvious first step. Ask before any Kaggle submission.

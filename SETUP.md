# Setting up genML

## What you need

- Linux or WSL2 (developed on WSL2 with Ubuntu 24.04), Python 3.10 to 3.12, git.
- Optional: an NVIDIA GPU. XGBoost and CatBoost train on it; LightGBM runs on the CPU. Everything,
  including the test suite, also works without one.
- For Kaggle problems: a Kaggle API token.
- For the autopilot:
  - Claude Code (`claude`), signed in to your Claude plan. The sessions bill the subscription; API
    keys are removed from their environment.
  - `bubblewrap` and `socat`. Agent sessions and agent-written code run in a sandbox; without these
    the autopilot refuses to start.
  - Optional: Codex (`codex`), signed in to a ChatGPT plan, if you want a different model family to give
    the second opinion on plans. It is off by default (the second opinion comes from Claude); to use it,
    put `codex-cli` first in `ORDER` in `src/genML/autopilot/second_opinion.py`.

## 1. Python environment (pip)

```bash
cd /path/to/genML
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -r requirements.txt -r requirements-dev.txt -r requirements-dashboard.txt
VIRTUAL_ENV=.venv uv pip install kagglehub kaggle "mcp>=2" jsonschema   # Kaggle, MCP server, answer checks
VIRTUAL_ENV=.venv uv pip install torch pytabkit tabpfn                  # optional: RealMLP, TabM, TabPFN
source .venv/bin/activate
```

The pip wheels of XGBoost and CatBoost include GPU support. `.venv` is gitignored; worktrees of the
autopilot link to it.

TabPFN's weights download only after you accept Prior Labs' license and set `TABPFN_TOKEN`. Download
them once outside the autopilot: pipeline runs of agent code have no network.

## 2. Optional: RAPIDS conda environment

`environment.yml` builds a conda environment with RAPIDS cuML (CUDA 11.8) for GPU versions of the
linear models and random forest. It is not needed for anything else.

```bash
mamba env create -f environment.yml
conda activate genml
python -c "import cuml; print(cuml.__version__)"
```

Keep `requirements.txt`, `pyproject.toml` and `environment.yml` in sync when dependencies change.

## 3. Kaggle access

Create a token at https://www.kaggle.com/settings/api, then either:

```bash
export KAGGLE_API_TOKEN=...                 # or
mkdir -p ~/.kaggle && echo "<token>" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

Never put the token in the repository. Data downloads to the kagglehub cache
(`~/.cache/kagglehub/competitions/<slug>/`) or `datasets/<name>/`; Kaggle data is never committed.

## 4. Autopilot prerequisites

```bash
sudo apt install bubblewrap socat     # the sandbox for agent sessions and agent code
claude                                # once, to sign in to your Claude plan
codex login                           # optional, only if Codex gives the second opinion
python src/genML/main.py --llm-status # shows which of claude-cli / codex-cli can answer
```

Before the first autopilot start on a problem, measure its run-to-run noise: `python src/genML/main.py
--baseline NAME` queues one run and a seed test, and the judge refuses to start until both have finished.

`--allow-unsandboxed` runs the autopilot without bubblewrap. Agent code then has your permissions
(credentials are still removed from its environment); use it only if you accept that.

Long runs should outlive the terminal or Claude Code session that started them. Start the job worker and
the autopilot as user services, for example:

```bash
systemd-run --user --setenv=PATH="$PATH" --unit genml-worker --working-directory=/path/to/genML \
  /path/to/genML/.venv/bin/python -m src.genML.jobs worker
```

and the autopilot itself:

```bash
systemd-run --user --setenv=PATH="$PATH" --unit genml-autopilot-fraud -- \
  /path/to/genML/.venv/bin/python src/genML/main.py --autopilot fraud-s6e11 --cycles 10
```

## 5. Check the installation

```bash
python src/genML/main.py --list-problems
pytest -m "not slow"                  # the fast suite, a few minutes
nvidia-smi                            # GPU visible (optional)
```

## Optional pieces

- **MCP server**: `claude mcp add genml -- /path/to/genML/.venv/bin/python -m src.genML.mcp_server`
- **Dashboard**: `python run_api.py`, then `cd dashboard && npm install && npm run dev`

## Troubleshooting

- **Out of GPU memory**: lower `budget.n_trials` or use fewer models; runs go through the job queue
  one at a time, so two runs never share the GPU.
- **A run or the job worker died**: `python -m src.genML.jobs list` shows jobs; `cancel JOB_ID` clears
  one. The autopilot resumes from its last step when started again.
- **The autopilot does not start**: it prints why (no sandbox tools, an invalid permission header, a run
  already in progress). `--autopilot-status NAME` shows the state of each cycle.

# 2Dattention Long-Term Memory

## Project

- Repository: `HUSRCF/2Dattention`
- Local root: `/home/husrcf/Code/2Dattention`
- Primary branch: `main`
- Purpose: a small PyTorch prototype for 2D spatial prefill memory and lattice Attention Residual reads.
- Core package: `src/attention2d/`
- Runnable experiments and data helpers: `scripts/`
- Research notes and evidence boundaries: `docs/`

## Current State

- The local checkout is complete after rsync and was clean before this memory file was added.
- Latest local commit at memory creation: `bbd0b0f Record RF-DETR seed43 full continuation result`.
- The GitHub repository at `https://github.com/HUSRCF/2Dattention` was checked on 2026-09-08 and was empty at that time.
- This checkout initially had no Git remote configured; `origin` should point to the GitHub URL when syncing.
- Generated `data/` and `results/` content is generally ignored by `.gitignore`; do not remove existing tracked artifacts without explicit instruction.

## Environment and Checks

- Expected environment: conda env `AIAA`, with PyTorch 2.8.0.
- Shape smoke test:
  `/opt/anaconda3/bin/conda run -n AIAA python scripts/run_shape_demo.py`
- Toy smoke test:
  `/opt/anaconda3/bin/conda run -n AIAA python scripts/run_toy_task.py --steps 30`
- Read `docs/review.md` before making broad performance claims; toy tasks are information-flow checks, not evidence of detector superiority.

## Working Conventions

- Keep changes focused and preserve user-generated research artifacts.
- Prefer existing scripts and documented protocols over ad hoc experiment changes.
- Before syncing: run `git status --short --branch`, inspect the diff, commit intentional changes, then push `main` to `origin`.
- Do not commit downloaded datasets, model weights, or generated outputs unless a task explicitly requires them.

## Resume Checklist

1. Read this file, `README.md`, and the relevant files under `docs/`.
2. Check `git status`, remotes, and the latest commit.
3. Confirm whether a requested experiment is a smoke test, a formal run, or analysis-only.
4. Record durable decisions and result interpretations here when they affect future work.

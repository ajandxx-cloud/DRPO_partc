<!-- GSD:project-start source:PROJECT.md -->
## Project

**TRC_DRT_DRPO_revision**

This is a manuscript revision project for a Transportation Research Part C submission on many-to-one demand-responsive transit with dynamic meeting-point recommendation and pricing. The project coordinates diagnosis, revision planning, and iterative manuscript edits for `part_c模版_DRT_0522_manuscript_working.md`, while preserving `part_c模版_DRT_0522_manuscript_original.md` as the untouched backup.

The manuscript's core method is DRPO: Dynamic Meeting-Point Recommendation and Pricing Optimization. DRPO must remain framed as `recommend -> predict -> price`, where the prediction module estimates option-level marginal insertion costs and service-time proxies, the pricing module converts predicted costs into option-specific price adjustments, and SPO/SPO+ aligns prediction with downstream pricing decisions.

**Core Value:** Raise the manuscript to the standard of a rigorous Transportation Research Part C submission through disciplined diagnosis, mathematically consistent revision, and one focused revision task per iteration.

### Constraints

- **Journal fit**: Revisions must target Transportation Research Part C standards for methodological clarity, modeling rigor, empirical credibility, and precise claims.
- **Manuscript safety**: `part_c模版_DRT_0522_manuscript_original.md` is a backup and must not be modified.
- **Iteration discipline**: Each revision iteration should complete exactly one focused task, avoiding sprawling edits.
- **Method preservation**: DRPO must remain `recommend -> predict -> price`; revisions should clarify rather than replace this identity.
- **Math consistency**: Notation, MDP state/action/reward, Bellman equations, marginal insertion cost approximation, pricing optimization, and SPO/SPO+ formulation must agree with each other.
- **Evidence alignment**: Experimental claims must be traceable to available tables, figures, logs, scripts, or explicitly identified missing evidence.
- **Workspace boundaries**: Writing tasks should primarily touch manuscript and revision artifacts; code/experiment tasks should remain scoped to `ooh_code/`, `experiments/`, `data/`, and generated outputs when needed.
- **Git state**: The root DRT repository was initialized for GSD planning. `ooh_code/` contains its own nested Git repository, so root-level planning commits should avoid accidentally treating implementation history as a normal subfolder.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.10 - Main simulation, model training, experiment orchestration, data generation, plotting, and manuscript table tooling under `ooh_code/`, `experiments/`, `figures/`, `data/`, and `related_work/gsd改公式尝试/`. `ooh_code/README.md` states Python 3.10 as the project runtime.
- PowerShell - Windows GPU/run wrappers under `ooh_code/scripts/*.ps1`, including `ooh_code/scripts/run_rc_full12_gpu.ps1` and `ooh_code/scripts/gpu_env_check.ps1`.
- Windows batch - Environment/install/background launch wrappers in `ooh_code/install.cmd` and `ooh_code/scripts/run_yanjiao_drpo_dspo_tuning_background.cmd`.
- LaTeX/BibTeX - Manuscript source and references in `manuscript/trc_latex/manuscript.tex`, `manuscript/trc_latex/references.bib`, and `ooh_code/Smart Predict-Then-Optimize for Dynamic Meeting-Point Recommendation and Pricing in Many-to-One Demand-Responsive Transit.tex`.
- JSON/YAML - Experiment configs and saved argument dumps use JSON and YAML via `ooh_code/configs/*.json` and `ooh_code/Src/config.py`.
- Markdown/text - Planning, reviewer response, related-work, and manuscript notes under `planning/`, `revision_notes/`, `related_work/`, and `notes/`.
## Runtime
- Python 3.10 for the documented environment in `ooh_code/README.md`.
- CUDA-capable PyTorch runtime is optional in code paths but required by strict GPU wrappers in `ooh_code/scripts/gpu_env_check.ps1`, `ooh_code/scripts/run_rc_full12_gpu.ps1`, and `ooh_code/scripts/run_single_train_gpu.ps1`.
- Several Windows scripts pin `C:/Users/39583/AppData/Local/Programs/Python/Python38/python.exe` for local GPU runs (`ooh_code/README.md`, `ooh_code/scripts/run_rc_full12_gpu.ps1`, `ooh_code/scripts/run_single_train_gpu.ps1`), so local execution currently mixes documented Python 3.10 guidance with a Python 3.8 GPU interpreter.
- pip - Install declared dependencies with `python -m pip install -r requirements.txt` from `ooh_code/README.md`.
- Lockfile: missing. No `poetry.lock`, `uv.lock`, `Pipfile.lock`, or conda environment file detected for the full workspace.
## Frameworks
- PyTorch `>=2.0.1` in `ooh_code/requirements.txt` - Neural network predictors, PPO actor/critic models, CUDA device selection, checkpoints, and optimizers in `ooh_code/Src/config.py`, `ooh_code/Src/Utils/Predictors.py`, `ooh_code/Src/Utils/Actor.py`, `ooh_code/Src/Utils/Critic.py`, and `ooh_code/Src/Algorithms/*.py`.
- hygese `~=0.0.0.8` in `ooh_code/requirements.txt` - Hybrid Genetic Search CVRP solver used by routing/reoptimization code in `ooh_code/Src/Algorithms/Baseline.py`, `ooh_code/Environments/OOH/env_utils.py`, and `ooh_code/Src/Utils/Utils.py`.
- NumPy `~=1.25.1` in `ooh_code/requirements.txt` - Core array, random sampling, distance matrix, and result serialization layer across `ooh_code/Src/`, `ooh_code/scripts/`, and `experiments/figure_scripts/`.
- SciPy `~=1.11.1` in `ooh_code/requirements.txt` - Scientific computing dependency declared for the simulation stack.
- unittest - Standard-library tests under `ooh_code/tests/test_rc_full12_oat.py`.
- pytest-compatible discovery - Tests are named `test_*.py` under `ooh_code/tests/`, but no `pytest` dependency or pytest config is declared in `ooh_code/requirements.txt`.
- argparse - CLI argument parsing for training and data builders in `ooh_code/Src/parser.py`, `ooh_code/scripts/build_nyc_tlc_pilot.py`, `ooh_code/scripts/generate_yanjiao_instance.py`, and experiment runners under `ooh_code/scripts/`.
- PowerShell run wrappers - GPU and long-run orchestration in `ooh_code/scripts/run_rc_full12_gpu.ps1`, `ooh_code/scripts/run_rc_full12_parallel2.ps1`, `ooh_code/scripts/run_yanjiao_drpo_dspo_tuning_background.ps1`, and related scripts.
- Matplotlib Agg backend - Headless figure generation in `ooh_code/Src/Utils/Utils.py`, `experiments/figure_scripts/*.py`, `figures/make_paper_spatial_layouts.py`, and `ooh_code/scripts/plot_yanjiao_case_basemap.py`.
- LaTeX toolchain - Manuscript compilation artifacts are present in `manuscript/trc_latex/manuscript.aux`, `manuscript/trc_latex/manuscript.bbl`, `manuscript/trc_latex/manuscript.blg`, `manuscript/trc_latex/manuscript.log`, and `manuscript/trc_latex/manuscript.out`.
## Key Dependencies
- `torch>=2.0.1` - Learning models, optimizers, CUDA selection, tensor computation, and model checkpointing in `ooh_code/Src/config.py`, `ooh_code/Src/Algorithms/`, and `ooh_code/Src/Utils/`.
- `numpy~=1.25.1` - Simulation state, distance matrices, adjacency matrices, `.npy` sidecars, sensitivity analysis, and plotting data in `ooh_code/Src/Utils/Utils.py`, `ooh_code/scripts/generate_yanjiao_instance.py`, and `experiments/figure_scripts/*.py`.
- `hygese~=0.0.0.8` - CVRP solving and HGS routing labels in `ooh_code/Src/Algorithms/Baseline.py`, `ooh_code/Environments/OOH/env_utils.py`, and `ooh_code/Src/Utils/Utils.py`.
- `pyyaml>=6.0` - Writes experiment arguments to `args.yaml` in `ooh_code/Src/config.py`.
- `matplotlib~=3.7.2` - Publication plots, training curves, maps, and result figures in `ooh_code/Src/Utils/Utils.py`, `ooh_code/plot_loss_curves.py`, `ooh_code/plot_sensitivity_main_figures.py`, `experiments/figure_scripts/*.py`, and `figures/*.py`.
- `pandas>=2.0` - NYC TLC parquet/GTFS processing in `ooh_code/scripts/build_nyc_tlc_pilot.py` and tabular analysis workflows.
- `pyarrow>=14.0` - Efficient parquet dataset filtering in `ooh_code/scripts/build_nyc_tlc_pilot.py`.
- `geopandas>=0.14` - Taxi-zone shapefile loading/projection in `ooh_code/scripts/build_nyc_tlc_pilot.py`.
- `contextily>=1.4.0` - Declared basemap/geospatial plotting dependency in `ooh_code/requirements.txt`; direct use was not detected in scanned Python files.
- `requests` - HTTP calls to Amap and map tile services in `ooh_code/scripts/fetch_yanjiao_full_data.py`, `ooh_code/scripts/fetch_yanjiao_poi_stats.py`, `ooh_code/scripts/plot_yanjiao_case_basemap.py`, `ooh_code/fig.10 beijing_base_map.py`, and `figures/12.16修改版本图片/Fig.10 Beijing_case/*.py`. This import is used but not declared in `ooh_code/requirements.txt`.
- `Pillow` (`PIL`) - Basemap tile image loading/enhancement in `ooh_code/scripts/plot_yanjiao_case_basemap.py`, `ooh_code/fig.10 beijing_base_map.py`, and `figures/12.16修改版本图片/Fig.10 Beijing_case/*.py`. This import is used but not declared in `ooh_code/requirements.txt`.
- `python-docx` (`docx`) - Generated Word tables and formula documents in `experiments/table_scripts/generate_result_tables.py`, `experiments/table_scripts/generate_beijing_table.py`, and `related_work/gsd改公式尝试/*.py`. This import is used but not declared in `ooh_code/requirements.txt`.
- `lxml` - Word XML/OMML manipulation in `related_work/gsd改公式尝试/create_omml_doc.py`, `related_work/gsd改公式尝试/generate_omml_final.py`, and `related_work/gsd改公式尝试/scripts/11_compact_xml.py`. This import is used but not declared in `ooh_code/requirements.txt`.
- `latex2mathml` - LaTeX-to-OMML formula conversion helper in `related_work/gsd改公式尝试/generate_omml_formulas.py`. This import is used but not declared in `ooh_code/requirements.txt`.
- `pywin32` (`win32com.client`) - Microsoft Word field-update automation in `related_work/gsd改公式尝试/scripts/10_field_update.py`. This import is used but not declared in `ooh_code/requirements.txt`.
## Configuration
- Primary runtime settings are CLI flags in `ooh_code/Src/parser.py`; `ooh_code/Src/config.py` copies parsed arguments into `Config`, sets seeds, creates experiment output directories, writes `args.yaml`, loads data, selects environment/algorithm classes dynamically, and selects CUDA or CPU.
- JSON calibration presets live in `ooh_code/configs/rc_main_pilot_3seed.json`, `ooh_code/configs/rc_main_pilot_3seed_mid_price.json`, `ooh_code/configs/rc_main_pilot_3seed_strong_price.json`, and `ooh_code/configs/rc_main_pilot_3seed_strong_price_u0m05.json`.
- Data-loading flags in `ooh_code/Src/parser.py` include `--instance`, `--data_seed`, `--data_seed_test`, `--n_passengers`, and `--yanjiao_prefix`; corresponding loaders are in `ooh_code/Src/Utils/Utils.py`.
- GPU configuration is controlled by `--gpu` in `ooh_code/Src/parser.py` and strict wrapper defaults in `ooh_code/scripts/gpu_env_check.ps1` and `ooh_code/scripts/run_single_train_gpu.ps1`.
- `.env` file present at `related_work/gsd改公式尝试/.env` - contents were not read.
- Dependency manifest: `ooh_code/requirements.txt`.
- Project layout guidance: `PROJECT_LAYOUT.md` and `ooh_code/PROJECT_LAYOUT.md`.
- No root `pyproject.toml`, `setup.py`, `tox.ini`, `pytest.ini`, `ruff.toml`, `.prettierrc`, or Node package manifest detected.
- PowerShell/batch execution wrappers: `ooh_code/scripts/*.ps1`, `ooh_code/scripts/*.cmd`, and `ooh_code/install.cmd`.
- LaTeX manuscript source and bibliography: `manuscript/trc_latex/manuscript.tex`, `manuscript/trc_latex/references.bib`, and `manuscript/trc_latex/drpo_trc_refs.bib`.
## Platform Requirements
- Use a Python virtual environment and install `ooh_code/requirements.txt` as documented in `ooh_code/README.md`.
- For main experiment runs, execute from `ooh_code/` so dynamic module loading in `ooh_code/Src/Utils/Utils.py` can build module paths containing `ooh_code`.
- Windows users can run GPU wrappers in `ooh_code/scripts/*.ps1`; those scripts assume a local Python path under `C:/Users/39583/AppData/Local/Programs/Python/Python38/python.exe`.
- CUDA-enabled PyTorch is required for strict GPU wrappers; `ooh_code/scripts/gpu_env_check.ps1` throws when CUDA is unavailable.
- Manuscript DOCX helpers require additional undeclared packages such as `python-docx`, `lxml`, `latex2mathml`, and, for Word COM field updates, `pywin32` plus Microsoft Word on Windows.
- Geospatial/real-data builders require undeclared or declared geospatial dependencies as used: `geopandas`, `pyarrow`, `pandas`, `requests`, and `Pillow`.
- Not applicable. This is a local research/manuscript workspace with experiment runners, generated result artifacts, and manuscript tooling; no deployable web service or hosted application target was detected.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Use lowercase snake_case for durable scripts and experiment runners: `ooh_code/sensitivity_analysis_dspo_plus_spo_oat.py`, `ooh_code/scripts/tune_rc_dspo_plus_lifted.py`, `figures/make_paper_spatial_layouts.py`.
- Legacy algorithm modules keep mixed-case class-matching filenames: `ooh_code/Src/Algorithms/DSPO.py`, `ooh_code/Src/Algorithms/DSPO_plus_SPO.py`, `ooh_code/Src/Algorithms/OnlinePricingSystem.py`, `ooh_code/Src/Utils/Utils.py`.
- Test files use `test_*.py`: `ooh_code/tests/test_dspo_plus_lifted_spo.py`, `ooh_code/tests/test_rc_full12_oat.py`.
- Figure and manuscript artifact names may contain spaces, punctuation, dates, and Chinese text. Keep new code out of artifact folders such as `figures/12.16修改版本图片/` unless the script specifically regenerates those artifacts.
- Use lowercase snake_case for new functions: `parse_args`, `build_cmd`, `parse_metrics`, `write_csv` in `ooh_code/scripts/tune_rc_dspo_plus_lifted.py`; `extract_data`, `parse_synthetic`, `plot_layout` in `figures/make_paper_spatial_layouts.py`.
- Preserve existing legacy method names when overriding or calling algorithm APIs: `get_action`, `update`, `reset`, `environment_parameters`, `DSPO_parameters` in `ooh_code/Src/parser.py`, `ooh_code/run.py`, and `ooh_code/Src/Algorithms/*.py`.
- Private helpers use a leading underscore when they are algorithm internals: `_safe_exp`, `_pricing_oracle_lifted_torch`, `_pricing_oracle_lifted_np` in `ooh_code/Src/Algorithms/DSPO_plus_SPO.py`.
- Use lowercase snake_case for local variables and data structures in new scripts: `run_id`, `output_dir`, `served_rate`, `factor_grid`, `stage2_jobs` in `ooh_code/sensitivity_analysis_dspo_plus_spo_oat.py`.
- Constants are uppercase at module scope: `METRIC_REGEX`, `SUMMARY_METRICS`, `RC_FULL12_FACTORS`, `PROFILE_LIBRARY` in `ooh_code/sensitivity_analysis_dspo_plus_spo_oat.py`; `ROOT`, `OUT` in `figures/make_paper_spatial_layouts.py`.
- Preserve domain terms and legacy field names where they are part of existing data contracts: `remainingCapacity`, `routePlan`, `incentiveSensitivity` in `ooh_code/Environments/OOH/containers.py`; `home_delivery_loc`, `accepted_price`, `route_data` in `ooh_code/run.py`.
- Use `PascalCase` for classes and dataclasses: `Config` in `ooh_code/Src/config.py`, `Parser` in `ooh_code/Src/parser.py`, `Fleet` and `Customer` in `ooh_code/Environments/OOH/containers.py`, `RunRecord` in `ooh_code/sensitivity_analysis_dspo_plus_spo_oat.py`.
- Use dataclasses for small structured records in newer code: `RunRecord` in `ooh_code/sensitivity_analysis_dspo_plus_spo_oat.py`, `SPOExperimentResult` in `ooh_code/Src/Algorithms/DSPO_plus_SPO.py`, `Location` and `Vehicle` in `ooh_code/Environments/OOH/containers.py`.
- Type hints are common in newer orchestration scripts and should be used for function signatures that pass records or paths: `build_cmd(args: argparse.Namespace, ...) -> List[str]` and `parse_metrics(log: Path) -> Optional[Dict[str, float]]` in `ooh_code/scripts/tune_rc_dspo_plus_lifted.py`.
## Code Style
- No formatter configuration is detected. There is no `pyproject.toml`, `.flake8`, `ruff.toml`, `setup.cfg`, `.prettierrc`, or `eslint.config.*` in the scanned repo root.
- Follow the dominant Python style manually: 4-space indentation, one top-level helper per logical operation, `if __name__ == "__main__": main()` for executable scripts, and concise blank-line separation around top-level constants/functions.
- For new code, prefer the cleaner style used in `ooh_code/scripts/tune_rc_dspo_plus_lifted.py`, `ooh_code/sensitivity_analysis_dspo_plus_spo_oat.py`, and `figures/make_paper_spatial_layouts.py` over older dense formatting in `ooh_code/run.py` and `data/run_baseline(1).py`.
- Keep Matplotlib scripts headless when they generate files: call `matplotlib.use('Agg')` before importing `matplotlib.pyplot`, as in `experiments/figure_scripts/plot_dspo_plus_training_curve.py` and `ooh_code/Src/Utils/Utils.py`.
- Use `Path` for path assembly in new scripts: `ROOT = Path(__file__).resolve().parent.parent` in `ooh_code/scripts/tune_rc_dspo_plus_lifted.py` and `OUT = ROOT / "manuscript" / "trc_latex" / "images"` in `figures/make_paper_spatial_layouts.py`.
- Not detected. No lint tool is configured or listed in `ooh_code/requirements.txt`.
- Because linting is not enforced, new code should self-enforce simple rules: avoid wildcard imports, avoid unused imports, keep generated artifact logic separate from reusable algorithms, and keep functions testable without starting training jobs.
## Import Organization
- No configured package alias system is detected.
- Local imports assume the process working directory is `ooh_code/`, so imports use root-relative package names such as `Src.*` and `Environments.*`: `ooh_code/run.py`, `ooh_code/tests/test_dspo_plus_lifted_spo.py`.
- Run tests and training from `ooh_code/` when using these imports. Running `python -m pytest ooh_code/tests -q` from repo root fails collection because `Src` is not importable.
## Error Handling
- Use explicit `RuntimeError` for failed subprocesses, invalid scan state, unavailable CUDA, and missing metrics in orchestration scripts: `ooh_code/sensitivity_analysis_dspo_plus_spo_oat.py`, `ooh_code/scripts/tune_rc_dspo_plus_lifted.py`, `ooh_code/scripts/compare_algorithms_dspo_vs_plus.py`.
- Use `ValueError` for invalid input files, missing data blocks, and impossible model/data states: `figures/make_paper_spatial_layouts.py`, `ooh_code/Src/Utils/Utils.py`, `ooh_code/Src/config.py`.
- Use narrow exception handling where parsing user/data input is expected to fail: `experiments/table_scripts/generate_result_tables.py` catches `(ValueError, TypeError)` in `_is_num`; `ooh_code/scripts/tune_rc_dspo_plus_lifted.py` catches `subprocess.TimeoutExpired` and retries.
- Return `None` for non-fatal missing parse results or infeasible oracle outputs, then assert/check at call sites: `parse_metrics` in `ooh_code/scripts/tune_rc_dspo_plus_lifted.py`; `_pricing_oracle_lifted_np` and `_pricing_oracle_lifted_torch` in `ooh_code/Src/Algorithms/DSPO_plus_SPO.py`.
## Logging
- Long training runs print progress and metrics, then `Config` redirects stdout through `Utils.Logger` based on `--log_output`: `ooh_code/Src/config.py`, `ooh_code/Src/Utils/Utils.py`, `ooh_code/run.py`.
- Orchestration scripts print bracketed status messages and flush during long subprocess runs: `[INFO]`, `[RUN]`, `[DONE]`, `[DRY-RUN]` in `ooh_code/scripts/tune_rc_dspo_plus_lifted.py` and `ooh_code/sensitivity_analysis_dspo_plus_spo_oat.py`.
- Table and figure scripts print saved output paths only after successful writes: `experiments/table_scripts/generate_beijing_table.py`, `figures/make_paper_spatial_layouts.py`, `experiments/figure_scripts/plot_dspo_plus_training_curve.py`.
- Use `encoding="utf-8", errors="ignore"` for reading potentially mixed-encoding logs and LaTeX/manuscript text: `ooh_code/scripts/tune_rc_dspo_plus_lifted.py`, `ooh_code/scripts/trc_review_workflow.py`, `figures/make_paper_spatial_layouts.py`.
## Comments
- Use comments for domain assumptions, experiment phase boundaries, data interpretations, and known compatibility decisions: outside-option probability notes in `ooh_code/tests/test_dspo_plus_lifted_spo.py`, phase boundaries in `experiments/figure_scripts/plot_dspo_plus_training_curve.py`, and alias compatibility in `ooh_code/Src/Algorithms/DRPO.py`.
- Keep comments close to the relevant data contract when indexing dense result tuples such as `stats[...]` in `ooh_code/run.py`.
- Avoid decorative separator comments in new reusable modules. They are common in one-off figure/manuscript scripts under `experiments/figure_scripts/` and `related_work/gsd改公式尝试/scripts/`, but they add noise in core algorithm code.
- Not applicable.
- Python docstrings are used in newer modules for public scripts/classes and important numerical helpers: `ooh_code/scripts/tune_rc_dspo_plus_lifted.py`, `ooh_code/Src/Algorithms/DSPO_plus_SPO.py`, `ooh_code/Src/Algorithms/DRPO.py`.
## Function Design
## Module Design
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## System Overview
```text
```
## Component Responsibilities
| Component | Responsibility | File |
|-----------|----------------|------|
| CLI parser | Defines experiment, environment, algorithm, pricing, routing, CNN, SPO+, and PPO flags. Use it as the source for supported command-line parameters. | `ooh_code/Src/parser.py` |
| Runtime config | Resolves output paths, seeds NumPy/Torch, loads datasets, constructs train/test environments, dynamically loads algorithms, and selects device/optimizer. | `ooh_code/Src/config.py` |
| Main solver | Runs the train/eval loop for non-PPO algorithms by alternating `model.get_action()`, `env.step()`, and `model.update()`. | `ooh_code/run.py` |
| PPO solver | Separate runner for PPO-specific action masking, actor/critic updates, and terminal route reward calculation. | `ooh_code/run_ppo.py` |
| DRT environment | Owns passenger arrivals, parcel point capacity, fleet route state, customer choice calls, route reoptimization, and per-step statistics. | `ooh_code/Environments/OOH/Parcelpoint_py.py` |
| Choice model | Implements offer and pricing MNL choice, outside option handling, Gumbel noise, travel-time utility, and optional sidecar utilities. | `ooh_code/Environments/OOH/customerchoice.py` |
| Routing helpers | Provides cheapest insertion, parcel point reset, HGS reoptimization, and fleet/container utilities. | `ooh_code/Environments/OOH/env_utils.py` |
| Algorithm base | Defines shared module movement, gradient clearing, saving, stepping, and reset hooks. | `ooh_code/Src/Algorithms/Agent.py` |
| Baseline policies | Implements static pricing and nearest-feasible offering baselines with terminal HGS route cost. | `ooh_code/Src/Algorithms/Baseline.py` |
| Heuristic policy | Implements heuristic insertion-cost pricing/offering and terminal HGS route cost. | `ooh_code/Src/Algorithms/Heuristic.py` |
| DSPO | Implements CNN/linear cost prediction, pricing/offering actions, training labels, travel-time prediction hooks, and HGS terminal labeling. | `ooh_code/Src/Algorithms/DSPO.py` |
| DRPO public entry | Exposes the DRPO algorithm name as a subclass of the legacy SPO+ implementation. | `ooh_code/Src/Algorithms/DRPO.py` |
| SPO+ implementation | Extends DSPO with lifted pricing oracles, SPO+ replay data, warmup/rampup weighting, global option labels, and mixed Huber/SPO+ updates. | `ooh_code/Src/Algorithms/DSPO_plus_SPO.py` |
| Predictors | Provides `LinReg`, `CNN_2d`, `CNN_TravelTime`, and experimental `CNN_3d` modules used by DSPO/DRPO. | `ooh_code/Src/Utils/Predictors.py` |
| Data utilities | Loads instance files, adjacency matrices, choice utility sidecars, dynamic modules, memory buffers, route extraction, and plotting/logging helpers. | `ooh_code/Src/Utils/Utils.py` |
| Batch experiment runners | Wrap `run.py` with `subprocess`, parse logs, persist CSV summaries, and write analysis outputs. | `ooh_code/scripts/run_main_pilot_3seed.py`, `ooh_code/scripts/run_all6_strategies.py`, `ooh_code/scripts/run_ablation_spo.py`, `ooh_code/scripts/run_joint_sensitivity.py` |
| Instance generators | Build Beijing/Yanjiao, bus-style, commuter-aware, and NYC pilot input files consumed by `Utils.load_demand_data()`. | `ooh_code/scripts/generate_yanjiao_instance.py`, `ooh_code/scripts/generate_yanjiao_commuter_instance.py`, `ooh_code/scripts/generate_yanjiao_bus_style_instance.py`, `ooh_code/scripts/build_nyc_tlc_pilot.py` |
| Figure/table builders | Convert logs and result files into publication figures and Word/LaTeX tables. | `experiments/figure_scripts/`, `experiments/table_scripts/`, `ooh_code/scripts/fill_manuscript_results.py` |
| Manuscript source | Contains the TRC LaTeX manuscript, reference database, and image references. | `manuscript/trc_latex/manuscript.tex`, `manuscript/trc_latex/drpo_trc_refs.bib`, `manuscript/trc_latex/images/` |
## Pattern Overview
- `ooh_code/run.py` is the primary runtime entry point for DRPO/DSPO/Baseline/Heuristic runs; scripts under `ooh_code/scripts/` call it as subprocesses instead of importing solver internals.
- `ooh_code/Src/config.py` binds CLI flags to concrete data, environment, algorithm class, output paths, device, and optimizer before `ooh_code/run.py` instantiates `Solver`.
- Algorithms share an `Agent` interface and are selected by string name through `Utils.dynamic_load()`.
- The environment mutates fleet, parcel point capacity, route data, and statistics across passenger arrivals; algorithms observe state and return prices/offers.
- Publication artifacts are generated outside the algorithm runtime through `experiments/` and manuscript-specific scripts.
## Layers
- Purpose: Separates code/experiment work from writing/manuscript work.
- Location: `PROJECT_LAYOUT.md`, `planning/代码和论文区分-避免混乱.md`
- Contains: Agent scope rules, top-level map, and workflow guidance.
- Depends on: Not applicable.
- Used by: Human and agent workflows that decide whether changes belong under `ooh_code/`, `experiments/`, `data/`, `manuscript/`, `figures/`, `revision_notes/`, or `planning/`.
- Purpose: Starts training/evaluation from CLI parameters.
- Location: `ooh_code/run.py`, `ooh_code/run_ppo.py`
- Contains: `Solver` classes, `train()`, `eval()`, `eval2()`, `main()`.
- Depends on: `ooh_code/Src/parser.py`, `ooh_code/Src/config.py`, `ooh_code/Src/Utils/Utils.py`.
- Used by: Direct CLI runs and subprocess orchestration scripts in `ooh_code/scripts/`.
- Purpose: Converts parsed flags into concrete runtime objects and filesystem paths.
- Location: `ooh_code/Src/config.py`, `ooh_code/Src/parser.py`, `ooh_code/configs/*.json`
- Contains: CLI options, experiment path derivation, data loading, dynamic environment/algorithm loading, GPU/CPU selection, optimizer selection, JSON experiment config payloads.
- Depends on: `ooh_code/Src/Utils/Utils.py`, `ooh_code/Environments/`, `ooh_code/Src/Algorithms/`.
- Used by: `ooh_code/run.py`, `ooh_code/run_ppo.py`, and batch runners.
- Purpose: Represents the online DRT process: passenger arrivals, options, choices, route state, capacity, and terminal cost evaluation.
- Location: `ooh_code/Environments/OOH/Parcelpoint_py.py`, `ooh_code/Environments/OOH/customerchoice.py`, `ooh_code/Environments/OOH/env_utils.py`, `ooh_code/Environments/OOH/containers.py`
- Contains: Mutable fleet/parcel-point/customer state, MNL choice, outside option, HGS route calls, dataclass-like containers.
- Depends on: `hygese`, NumPy, `ooh_code/Src/Utils/Utils.py`.
- Used by: `Config.get_domain()` and all algorithms through `env.step()`.
- Purpose: Produces offer/pricing decisions and learns cost/travel-time predictors.
- Location: `ooh_code/Src/Algorithms/`
- Contains: `Agent`, `Baseline`, `Heuristic`, `DSPO`, `DRPO`, `DSPO_plus_SPO`, `PPO`, menu-selection variants, `OnlinePricingSystem`.
- Depends on: `ooh_code/Src/Utils/`, `ooh_code/Environments/`, Torch, SciPy Lambert W, HGS solver.
- Used by: `Config.algo` and `Solver.model`.
- Purpose: Provides data loading, dynamic imports, state encodings, replay buffers, route helpers, logging, plotting, and neural network predictors.
- Location: `ooh_code/Src/Utils/`
- Contains: `Utils.py`, `Predictors.py`, `Actor.py`, `Critic.py`, `Basis.py`.
- Depends on: NumPy, Torch, Matplotlib, HGS, local environment containers.
- Used by: `Config`, algorithms, environments, and runners.
- Purpose: Stores raw/prepared instances, generated adjacency matrices, experiment logs, checkpoints, and analysis CSVs.
- Location: `ooh_code/Environments/OOH/*`, `ooh_code/data/`, `ooh_code/Experiments/`, `data/`
- Contains: Amazon, Homberger-Gehring, Beijing bus, Beijing Yanjiao, NYC pilot instance files, raw transit archives, shapefiles, logs, generated experiment outputs.
- Depends on: File naming conventions in `Utils.load_demand_data()`.
- Used by: `Config`, `Utils.load_demand_data()`, batch scripts, plotting/table scripts.
- Purpose: Converts experiment outputs into figures, Word tables, manuscript text, and LaTeX build artifacts.
- Location: `experiments/`, `figures/`, `manuscript/`, `revision_notes/`, `related_work/`, `notes/`
- Contains: Figure scripts, table scripts, image files, manuscript source, review notes, draft text, literature notes.
- Depends on: Result logs and CSVs under `ooh_code/Experiments/` plus manuscript image/table naming.
- Used by: Writing workflows and manuscript assembly.
## Data Flow
### Primary Request Path
### Evaluation Path
### Batch Experiment Path
### Data Loading Path
### Publication Path
- Runtime state is mutable and episode-scoped: `Parcelpoint_py` owns `fleet`, `parcelPoints`, `data`, `steps`, `service_time`, `total_prices`, `total_discounts`, and `quit_count` (`ooh_code/Environments/OOH/Parcelpoint_py.py:122`, `ooh_code/Environments/OOH/Parcelpoint_py.py:291`).
- Algorithm training state lives on algorithm objects: replay buffers, neural modules, terminal label caches, and SPO+ training dictionaries (`ooh_code/Src/Algorithms/DSPO.py:15`, `ooh_code/Src/Algorithms/DSPO_plus_SPO.py:50`).
- Output state is filesystem-based under `ooh_code/Experiments/`, `figures/`, and `manuscript/tables/`.
## Key Abstractions
- Purpose: Gives algorithms a common module lifecycle and optimizer API.
- Examples: `ooh_code/Src/Algorithms/Agent.py`, `ooh_code/Src/Algorithms/DSPO.py`, `ooh_code/Src/Algorithms/Baseline.py`
- Pattern: Subclasses implement policy-specific `get_action*()` and `update()` methods while reusing `init()`, `save()`, `step()`, and `reset()`.
- Purpose: Represents one online DRT decision epoch with current customer, fleet, parcel points, and step index.
- Examples: `ooh_code/Environments/OOH/Parcelpoint_py.py`, `ooh_code/Environments/OOH/containers.py`
- Pattern: `make_state()` returns `[newCustomer, fleet, parcelPoints, steps]`; algorithms must treat this shape as the state contract.
- Purpose: Keep location, parcel point, vehicle, fleet, and customer attributes accessible with both attribute and dict-like indexing.
- Examples: `ooh_code/Environments/OOH/containers.py`
- Pattern: `Location`, `ParcelPoint`, `ParcelPoints`, `Vehicle`, `Fleet`, and `Customer` are lightweight records; `__getitem__` returns attributes for legacy code compatibility.
- Purpose: Allows `--algo_name` and `--env_name` strings to select implementation classes.
- Examples: `ooh_code/Src/Utils/Utils.py:154`, `ooh_code/Src/config.py:89`, `ooh_code/Src/config.py:123`
- Pattern: Use class/file names that match parser choices and searchable filenames; avoid adding algorithms whose class name does not match the intended `--algo_name`.
- Purpose: Convert grid-encoded fleet/customer state and capacity into cost/travel-time predictions.
- Examples: `ooh_code/Src/Utils/Predictors.py`, `ooh_code/Src/Algorithms/DSPO.py`
- Pattern: `CNN_2d` and `LinReg` accept `(features, capacity)`; `CNN_TravelTime` subclasses `CNN_2d`.
- Purpose: Stores fixed-shape tensors for supervised DSPO/DRPO updates.
- Examples: `ooh_code/Src/Utils/Utils.py:786`
- Pattern: Preallocate tensors on `config.device` with dimensions derived from `grid_dim`, `n_input_layers`, and `buffer_size`.
- Purpose: Implements deployed clipped Lambert-W pricing and lifted SPO+ decision objects.
- Examples: `ooh_code/Src/Algorithms/DSPO.py:165`, `ooh_code/Src/Algorithms/DSPO_plus_SPO.py:110`, `ooh_code/Src/Algorithms/DSPO_plus_SPO.py:193`
- Pattern: Keep deployment pricing and SPO+ training oracle aligned when changing pricing behavior.
- Purpose: Allows each named instance to supply coordinates, distance/duration matrix, adjacency, service times, metadata, and optional choice utility.
- Examples: `ooh_code/Src/Utils/Utils.py:442`, `ooh_code/Environments/OOH/Beijing_Yanjiao/`
- Pattern: Files are named with a base prefix plus sidecar suffixes like `_coords.txt`, `_duration_matrix.txt`, `_metadata.json`, `_adjacency{k}.npy`, and `_choice_utility.npy`.
## Entry Points
- Location: `ooh_code/run.py`
- Triggers: `python run.py ...` from `ooh_code/` or batch scripts.
- Responsibilities: Parse args, build config, instantiate `Solver`, train, evaluate, and print metrics.
- Location: `ooh_code/run_ppo.py`
- Triggers: `python run_ppo.py ...` when using PPO.
- Responsibilities: PPO-specific action masking, actor/critic update loop, and terminal route reward.
- Location: `ooh_code/scripts/run_main_pilot_3seed.py`
- Triggers: `python scripts/run_main_pilot_3seed.py ...` from `ooh_code/`.
- Responsibilities: Load JSON config, run DSPO/DRPO/static pricing jobs, parse logs, and persist `pilot_raw.csv`/`pilot_summary.csv`.
- Location: `ooh_code/scripts/run_all6_strategies.py`
- Triggers: `python scripts/run_all6_strategies.py ...`.
- Responsibilities: Run benchmark strategies through `run.py` and write analysis outputs.
- Location: `ooh_code/scripts/run_ablation_spo.py`
- Triggers: `python scripts/run_ablation_spo.py ...`.
- Responsibilities: Compare DRPO-Huber and DRPO variants using SPO loss weight differences.
- Location: `ooh_code/scripts/run_joint_sensitivity.py`
- Triggers: `python scripts/run_joint_sensitivity.py ...`.
- Responsibilities: Run parameter-grid comparisons and write sensitivity outputs.
- Location: `ooh_code/scripts/generate_yanjiao_instance.py`, `ooh_code/scripts/generate_yanjiao_commuter_instance.py`, `ooh_code/scripts/generate_yanjiao_bus_style_instance.py`, `ooh_code/scripts/build_nyc_tlc_pilot.py`
- Triggers: Manual CLI calls before experiment runs.
- Responsibilities: Produce dataset sidecars consumed by `Utils.load_demand_data()`.
- Location: `experiments/figure_scripts/`, `ooh_code/scripts/plot_yanjiao_case_layout.py`, `ooh_code/scripts/plot_yanjiao_case_academic.py`
- Triggers: Manual CLI/script execution.
- Responsibilities: Read result files and write PNG/PDF/SVG assets to `figures/` or manuscript image directories.
- Location: `experiments/table_scripts/generate_result_tables.py`, `experiments/table_scripts/generate_beijing_table.py`, `ooh_code/scripts/generate_yanjiao_table.py`
- Triggers: Manual CLI/script execution.
- Responsibilities: Read logs/CSVs and generate Word/LaTeX-ready tables under `manuscript/tables/` or analysis outputs.
- Location: `manuscript/trc_latex/manuscript.tex`
- Triggers: LaTeX build tools.
- Responsibilities: Assemble paper text, images, tables, appendix, and bibliography.
## Architectural Constraints
- **Threading:** Main simulation is a single-process, sequential episode loop in `ooh_code/run.py`; batch wrappers parallelize only by launching separate `run.py` subprocesses when script logic does so.
- **Global state:** `Config` redirects `sys.stdout` to `Utils.Logger` (`ooh_code/Src/config.py:45`, `ooh_code/Src/Utils/Utils.py:24`); code that imports `Config` in-process inherits this stdout mutation.
- **Randomness:** `Config` seeds NumPy and Torch with `args.seed` (`ooh_code/Src/config.py:21`), and `Parcelpoint_py.reset()` reseeds NumPy from `self.seed_value` when present (`ooh_code/Environments/OOH/Parcelpoint_py.py:122`).
- **Dynamic imports:** `Utils.dynamic_load()` searches recursively and skips `Experiments`; class names and filenames must remain discoverable under `ooh_code/Src/Algorithms/` or `ooh_code/Environments/` (`ooh_code/Src/Utils/Utils.py:154`).
- **Working directory:** Batch scripts build commands like `python run.py` and set `cwd` to the `ooh_code/` root; run them from or with paths relative to `ooh_code/` (`ooh_code/scripts/run_main_pilot_3seed.py:160`, `ooh_code/scripts/run_main_pilot_3seed.py:297`).
- **Filesystem outputs:** `Config` creates result directories and writes `args.yaml` under `ooh_code/Experiments/{env}/pricing|offering/{algo}/{experiment_suffix}/` (`ooh_code/Src/config.py:33`).
- **Data file naming:** New dataset variants must satisfy the sidecar naming used by `Utils.load_demand_data()` (`ooh_code/Src/Utils/Utils.py:442`).
- **Route solver dependency:** HGS/Hygese is initialized in environment and algorithm helpers; terminal route costs and labels depend on HGS-compatible matrices and nontrivial route sizes (`ooh_code/Environments/OOH/env_utils.py`, `ooh_code/Src/Algorithms/DSPO.py:1239`).
- **Circular imports:** No explicit circular chain was detected in the mapped runtime path. Algorithms depend downward on `Src.Utils` and `Environments`; `Config` dynamically imports algorithms/environments.
- **Non-source footprint:** `ooh_code/.claude/worktrees/`, `ooh_code/Experiments/`, `related_work/*/unpacked*`, generated Office unpack folders, PDFs, zips, and raw shapefiles are artifact/archive areas; do not treat them as canonical source modules.
## Anti-Patterns
### Editing Generated or Archive Trees
### Bypassing the Parser/Config Contract
### Changing Pricing Logic in Only One Place
### Reading Metrics from Console Text Without Stable Patterns
## Error Handling
- Missing dataset files raise `ValueError` from `Utils.load_demand_data()` (`ooh_code/Src/Utils/Utils.py:442`).
- Missing dynamic class/module resolution raises `ValueError` after printing a traceback in `Utils.dynamic_load()` (`ooh_code/Src/Utils/Utils.py:154`).
- Small or empty route instances return `0.0` from `Parcelpoint_py.reopt_for_eval()` instead of calling HGS (`ooh_code/Environments/OOH/Parcelpoint_py.py:234`).
- Batch scripts convert failed subprocesses/timeouts into row status fields and persisted summaries (`ooh_code/scripts/run_main_pilot_3seed.py:297`, `ooh_code/scripts/run_main_pilot_3seed.py:307`).
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->

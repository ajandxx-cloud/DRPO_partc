# DRT Project Layout

This project is split into code/experiment work and paper/writing work so parallel agents do not edit the same files.

## Code Agent Scope

Code agents may modify:

- `ooh_code/`: model code, algorithms, training runs, tests, experiment runners.
- `experiments/`: project-level figure/table generation scripts.
- `data/`: raw or prepared data files, when the task explicitly needs data changes.

Code agents should not modify:

- `manuscript/`
- `references/`
- `related_work/`
- `revision_notes/`
- `notes/`
- `planning/`

Generated figures should be written to `figures/`. Generated manuscript tables should be written to `manuscript/tables/`.

## Writing Agent Scope

Writing agents may modify:

- `manuscript/`: manuscript drafts, latest versions, templates, and generated tables.
- `figures/`: publication and presentation figures.
- `notes/`: writing notes and technical notes for the paper.
- `references/`: reference papers and literature notes.
- `related_work/`: related-work drafts, formula revisions, appendix material.
- `revision_notes/`: reviewer-response and paragraph revision material.
- `planning/`: workflow notes, prompts, and task separation instructions.

Writing agents should not modify:

- `ooh_code/`
- `experiments/`
- `data/`

## Current Top-Level Map

- `ooh_code/`: main implementation repository for DRPO/DSPO/SPO experiments.
- `experiments/figure_scripts/`: root-level plotting scripts moved out of the project root.
- `experiments/table_scripts/`: result-table generation scripts.
- `figures/`: generated figures for paper/PPT.
- `manuscript/`: paper drafts and latest manuscript files.
- `manuscript/tables/`: generated Word result tables.
- `notes/`: technical and writing notes.
- `planning/`: agent workflow and task-planning notes.
- `references/`: literature PDFs and summaries.
- `related_work/`: older related-work and formula-revision work products.
- `revision_notes/`: review and paragraph-revision material.

## Suggested Agent Prompts

Code thread:

```text
Only modify files under ooh_code/, experiments/, and data/ when necessary.
Do not touch manuscript/, figures/, references/, related_work/, revision_notes/, notes/, or planning/.
```

Writing thread:

```text
Only modify files under manuscript/, figures/, notes/, references/, related_work/, revision_notes/, and planning/.
Do not touch ooh_code/, experiments/, or data/.
```

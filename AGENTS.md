# AGENTS.md

Instructions for AI agents working in this repository. Setup, tests and lint are in [CONTRIBUTING.md](.github/CONTRIBUTING.md); this file covers only what keeps the docs correct.

## Update the docs in the same change

A change to the behavior of `packages/overture_matchers` updates the doc that describes it, in the same pull request. Find the doc from the code touched:

| Code changed | Doc to update |
| --- | --- |
| Job params in `core/config.py`: a new, renamed or dropped param, a default, a required flag, path rewriting | The params table in `packages/overture_matchers/docs/run-reference.md` |
| A stage starts or stops reading a param | That param's row in `docs/run-reference.md` |
| Run folder layout or stage markers in `core/io.py` | The layout block in `docs/run-reference.md`, and `docs/reruns.md` |
| `run_stage` or the rerun guard in `core/stages.py` and `core/config.py` | `docs/reruns.md`, including its diagram |
| A job's stages, outputs, required params or extra Spark packages | That job's page, for example `docs/places.md`, including its stage table and diagram |
| A new job or theme | A new reference page in `docs/`, a row in the README's Jobs table, and rows in `docs/README.md` and the README's Documentation table |
| A new module | A row in the README's Modules table |
| Install or run commands, or the pinned Spark and Sedona versions in `pyproject.toml` | The Install and Run sections of `packages/overture_matchers/README.md` |

A change with no behavior a reader can observe needs no doc edit. Do not add one to show the docs were considered.

## Rules for the package README

`packages/overture_matchers/README.md` is the page PyPI displays.

- Keep it an overview: description, install, run, the Modules and Jobs tables, and links. Detail goes in `packages/overture_matchers/docs/`.
- Use absolute `https://` links only. PyPI does not resolve relative links.
- No Mermaid diagrams and no GitHub admonitions (`> [!NOTE]`). PyPI renders neither.
- A Modules row describes the module's scope, not an inventory of its functions, so it stays true as the module grows.
- Any edit to this file releases a patch version on merge, because it is in the package's release path filters. Edits under `docs/` do not.

## Rules for every doc

- Document only what the code does. Read the source for each claim before writing it, and never describe planned behavior as present.
- Write in the present tense about the code as it is. No history, no roadmap, and no Status sections: the `Development Status` classifier in `pyproject.toml` carries the package's maturity.
- Stay platform agnostic. Describe what the package does on any Spark platform, and leave one platform's deployment detail to the repository that deploys it.
- Give each page in `docs/` one purpose: steps for a task (how-to), facts to look up (reference), or reasoning (explanation). Split a page that starts to serve two, and list every page under its group in `docs/README.md`.
- Open each Mermaid flowchart with the theme directive the existing diagrams use, and state its takeaway in the sentence above it.

## Check before finishing

Run from `packages/overture_matchers` after editing the package README. Both built files must report `PASSED`:

```sh
uv build --out-dir /tmp/overture-matchers-dist
uvx twine check /tmp/overture-matchers-dist/*
```

After editing a page in `docs/`, confirm that every relative link in it resolves to a file, and that no other doc still links to a page that was renamed.

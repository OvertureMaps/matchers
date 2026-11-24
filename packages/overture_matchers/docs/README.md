# overture-matchers docs

Technical detail for the [overture-matchers](../README.md) package. PyPI displays the package README, so the README stays an overview. Longer material belongs in these pages.

## How-to

| Page | Covers |
| --- | --- |
| [Package README](../README.md#install) | Installing the package and starting a run. |

## Reference

| Page | Covers |
| --- | --- |
| [run-reference.md](run-reference.md) | Job params and the run folder layout, shared by every job. |
| [places.md](places.md) | The places job: its params, stages and outputs. |

## Explanation

| Page | Covers |
| --- | --- |
| [reruns.md](reruns.md) | Why runs resume, how stage markers decide what runs, and why changed params need a new folder. |
| [python-migration.md](../../../docs/python-migration.md) | The repo-wide design for the port from Scala. |

## Adding a page

Each page serves one need. How-to pages give the steps for tasks, reference pages list facts to look up, and explanation pages give the reasoning. Add one reference page per theme job as the jobs are ported, and split pages that start to serve two needs.

Keep every page platform agnostic: describe what the package does on every Spark platform, and leave platform-specific deployment detail to the repositories that deploy the package.
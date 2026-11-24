# overture-matchers docs

Technical detail for the [overture-matchers](../README.md) package. The package README stays an overview, because PyPI displays it. Anything longer belongs here.

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
| [reruns.md](reruns.md) | Why a run resumes, how stage markers decide what runs, and why changed params need a new folder. |
| [python-migration.md](../../../docs/python-migration.md) | The repo-wide design for the port from Scala. |

## Adding a page

Each page serves one need: a how-to gives the steps for a task, a reference page lists facts to look up, and an explanation page gives the reasoning. Add a reference page per theme job as it is ported, and split a page that starts to serve two needs.

Keep every page platform agnostic: describe what the package does on any Spark platform, and leave the deployment detail of one platform to the repository that deploys it.
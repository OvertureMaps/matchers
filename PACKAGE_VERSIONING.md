# Package versioning and publishing

Each package under [`packages/`](packages) is versioned and released independently and automatically, following [OvertureMaps/overture-core](https://github.com/OvertureMaps/overture-core/blob/main/PACKAGE_VERSIONING.md).

## Version

Each package's `pyproject.toml` declares `dynamic = ["version"]`. [`setuptools-scm`](https://setuptools-scm.readthedocs.io/) resolves the version at build time from the package's latest `<package_dir>-v<version>` git tag, for example `overture_matchers-v0.2.0`.

## Bump

[python-semantic-release](https://python-semantic-release.readthedocs.io/) (PSR) reads the Conventional Commits since a package's last tag and picks the bump:

| Commit prefix                              | Bump  |
| ------------------------------------------ | ----- |
| `feat!: ...` / a `BREAKING CHANGE:` footer | major |
| `feat: ...`                                | minor |
| any other type (`fix`, `chore`, `docs`...) | patch |

PRs are squash-merged, so the PR title is the commit PSR reads. PSR attributes a commit to a package by the files it touched (`path_filters`), not by its scope. Only files that ship in the package count (its source, `pyproject.toml`, `README.md` and `LICENSE`), so changes to tests or `uv.lock` alone release nothing.

## Release

On push to `main`, [`publish_packages.yml`](.github/workflows/publish_packages.yml) runs three jobs for each package the push touched:

1. `release` runs PSR. If a bump is warranted, PSR tags `<package_dir>-v<version>` and publishes a GitHub Release with the notes; nothing is committed back to `main`. The tag is pushed with an overture-pull-requester App token, since `GITHUB_TOKEN` can't push tags that change workflows. The App key and the AWS credentials that fetch it are cleared before PSR runs, and no other job sees them.
2. `build`, with no secrets, builds only if `HEAD` carries the package's release tag: `uv build`, then `twine check --strict`, then upload `dist/` as an artifact. Because it keys on the tag, re-running a failed run publishes a release that was tagged but never uploaded.
3. `publish`, with only `id-token: write`, uploads the artifact to PyPI by trusted publishing and skips versions already there.

A manual run (`workflow_dispatch`) is a smoke test: no tag or release, every package builds from its current state and publishes to [Test PyPI](https://test.pypi.org/).

## Trusted publishing setup (one-time per package)

On both [pypi.org](https://pypi.org/manage/account/publishing/) and [test.pypi.org](https://test.pypi.org/manage/account/publishing/), add a trusted publisher (a pending one reserves the name before the first upload) for this repo and workflow `publish_packages.yml`, with environment `pypi-<package_dir>` on PyPI and `testpypi-<package_dir>` on Test PyPI. GitHub creates the environments on first use.

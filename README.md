# Overture Matchers

Python matchers for conflating Overture Maps data: places, divisions, and addresses.

Each matcher runs as a PySpark job on AWS Glue 5 (Spark 3.5, Python 3.11). It matches incoming records against the Overture baseline and assigns GERS IDs. This repo is a port of Overture's Scala matchers; the plan and progress are tracked in [#1](https://github.com/OvertureMaps/matchers/issues/1).

The matcher package, [`overture-matchers`](packages/overture_matchers), lives under [`packages/`](packages) and is published to PyPI.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

## Maintainers

This repository uses `MAINTAINERS.md` files to track ownership for [LFX Insights](https://insights.linuxfoundation.org/docs/introduction/maintainers/) ingestion. LFX scans the full repository tree, so these files can live anywhere.

To add a `MAINTAINERS.md` for a module or package, create it in the relevant directory using this format:

```markdown
# MAINTAINERS

| Name     | GitHub Username | Role            | Affiliation |
| -------- | --------------- | --------------- | ----------- |
| Jane Doe | @janedoe        | Lead Maintainer | Org         |
```

## License

[MIT](LICENSE)

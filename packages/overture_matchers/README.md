# overture-matchers

Python matchers for conflating Overture Maps data: places, divisions, and addresses.

Each matcher runs as a PySpark job on AWS Glue 5.0 (Spark 3.5, Python 3.11), matching incoming records against the Overture baseline and assigning GERS IDs. PySpark is not a runtime dependency; the Glue runtime provides it.

Source, issues and contributing guide: [OvertureMaps/matchers](https://github.com/OvertureMaps/matchers).

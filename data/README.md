# Local Data

The source datasets are intentionally excluded from this public repository.

Place the input file below in this directory before running the model or local
services:

- `total_consumption.csv`

Required columns:

- `timestamp`: timestamp at 15-minute intervals
- `consumption`: total electricity consumption for the interval

Optional source data such as `by_meter.csv` and `total_consumption.xlsx` is also
kept local and is not required by the scheme 1 serving path.

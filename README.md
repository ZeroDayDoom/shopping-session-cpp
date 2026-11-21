# Shopping Sessions — C++17 pipeline

From-scratch C++ implementation of `rules.md` (sessionization, attribution,
two Parquet outputs). Built for the competition scoring: read `DATA_DIR`,
write `OUTPUT_DIR/{user_daily,campaign_daily}` as Parquet readable by Polars.

## Design

- **Parse**: one streaming AVX2 pass per file chunk — `movemask` scans for
  `,`/`"`/`\n` drive the spec CSV state machine (quotes wrap whole fields
  only; wrong field count / stray quote / empty line = malformed; `\r` is
  data). `\n`-aligned chunks keep files parallel across cores.
- **Hashing**: two independent multiply-stream 128-bit keys (event ids,
  click ids, products, campaigns) finalized with mix64.
- **Users**: lock-free open-addressing table (dense atomic gids) with a
  16 MB direct-mapped L3-resident cache in front.
- **Dedup**: 128 hash buckets; per-bucket open-addressing maps with
  generation tags (no re-zeroing), thread-local reusable arenas. Dedup keeps
  the smallest `ingest_ts` per event/click id; the winner is validated after
  (a bad duplicate poisons its id).
- **Ordering**: events/clicks partitioned by root into bins, then
  per-bin sorts (cache-resident) fused with sessionization.
- **Attribution**: per-root two-pointer sweep with an active-campaign set
  (7-day windows), thread-tiled aggregators.
- **Writer**: hand-rolled minimal Parquet — PLAIN encoding, uncompressed,
  Thrift-compact footer, single data page per column.

## Build

```sh
make            # g++ -O3 -march=x86-64-v3
```

Docker (two-stage, runs as uid 1000, no network needed at runtime):

```sh
docker build -t sessions .
```

## Run

```sh
DATA_DIR=/data OUTPUT_DIR=/out ./pipeline
```

`NCPU` overrides the thread count (defaults to `hardware_concurrency`).

## Verification

- 36/36 edge-case fixtures (malformed lines, quoting, duplicates, ties,
  links, window edges, boundary timestamps).
- Exact match on the 1M-event dev set and the independent reference answers.

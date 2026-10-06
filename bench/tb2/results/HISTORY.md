| Date | Run | Policy | Tasks | Pass | Tokens (in / out) | Median agent | Note |
|---|---|---|---|---|---|---|---|
| 2026-10-04 18:33 | 20261004-1833-d66ead1 | e381398f | 81b4e82bd576 | 1/2 (50.0%) | 822381 / 40467 | 523.1s | tb2 smoke |
| 2026-10-04 18:54 | 20261004-1853-de4570e | e381398f | c7bd70e76232 | 11/14 (78.6%) ⚠1 infra | 6132368 / 246882 | 288.7s | tb2 v1 baseline (15-task subset) |
| 2026-10-05 19:39 | 20261005-1939-0fc05ad | e381398f | d4fbd5cc9a7b | 6/12 (50.0%) | 8850512 / 399008 | 840.5s | tb2-hard v1 baseline |
| 2026-10-05 21:58 | 20261005-2157-6956a0c | 78970755 | d4fbd5cc9a7b | 8/12 (66.7%) | 12195945 / 354384 | 840.3s | tb2-hard r1: --deadline effort pacing + SIGINT stop (feal+mcmc re-run via harbor jobs resume after a PC shutdown) |
| 2026-10-06 08:22 | 20261006-0822-7b23f29 | 2041ea7b | d4fbd5cc9a7b | 6/12 (50.0%) | 8178069 / 283601 | 841.2s | tb2-hard r2: per-call thinking cap + effort/think_ms logging |

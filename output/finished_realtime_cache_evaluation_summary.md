# Finished Match Evaluation From Realtime Cache

只使用每场开赛前最后一次实时缓存。

## Overall

| Matches | Outcome hit | Top1 bucket hit | Top2 bucket hit | Any exact score | Any score bucket | Mean deviation | Median deviation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 79 | 67.1% | 32.9% | 69.6% | 35.4% | 77.2% | 0.934 | 0.700 |

## Score Columns

| Score | Available | Exact | Outcome hit | Bucket hit | Mean deviation | Median deviation |
|---|---:|---:|---:|---:|---:|---:|
| model | 79 | 13.9% | 67.1% | 32.9% | 0.829 | 0.667 |
| aggressive | 79 | 13.9% | 67.1% | 36.7% | 1.020 | 0.600 |
| market | 79 | 8.9% | 59.5% | 30.4% | 0.949 | 0.667 |
| upset | 71 | 5.6% | 21.1% | 42.3% | 0.935 | 1.000 |

## Skipped

| Date | Match | Reason |
|---|---|---|
| 2026-06-12 00:00 | Mexico vs South Africa | no pre-match realtime cache |
| 2026-06-12 06:00 | South Korea vs Czechia | no pre-match realtime cache |
| 2026-06-13 03:00 | Canada vs Bosnia and Herzegovina | no pre-match realtime cache |
| 2026-06-13 09:00 | United States vs Paraguay | no pre-match realtime cache |
| 2026-06-14 03:00 | Qatar vs Switzerland | no pre-match realtime cache |
| 2026-06-14 06:00 | Brazil vs Morocco | no pre-match realtime cache |
| 2026-06-14 09:00 | Haiti vs Scotland | no pre-match realtime cache |
| 2026-06-14 12:00 | Australia vs Turkey | no pre-match realtime cache |
| 2026-06-15 01:00 | Germany vs Curaçao | no pre-match realtime cache |
| 2026-06-15 04:00 | Netherlands vs Japan | no pre-match realtime cache |
| 2026-06-15 07:00 | Ivory Coast vs Ecuador | no pre-match realtime cache |
| 2026-06-15 10:00 | Sweden vs Tunisia | no pre-match realtime cache |
| 2026-06-16 01:00 | Spain vs Cape Verde | no pre-match realtime cache |
| 2026-06-16 06:00 | Belgium vs Egypt | no pre-match realtime cache |
| 2026-06-16 06:00 | Saudi Arabia vs Uruguay | no pre-match realtime cache |
| 2026-06-16 12:00 | Iran vs New Zealand | no pre-match realtime cache |
| 2026-06-17 03:00 | France vs Senegal | no pre-match realtime cache |
| 2026-06-17 06:00 | Iraq vs Norway | no pre-match realtime cache |
| 2026-06-17 09:00 | Argentina vs Algeria | no pre-match realtime cache |
| 2026-06-17 12:00 | Austria vs Jordan | no pre-match realtime cache |
| 2026-06-18 01:00 | Portugal vs DR Congo | no pre-match realtime cache |
| 2026-06-18 04:00 | England vs Croatia | no pre-match realtime cache |
| 2026-06-18 07:00 | Ghana vs Panama | no pre-match realtime cache |
| 2026-06-18 10:00 | Uzbekistan vs Colombia | no pre-match realtime cache |
| 2026-07-19 05:00 | France vs England | no pre-match realtime cache |

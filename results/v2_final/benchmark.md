# Benchmark: `results/v2_final/gin_rummy_bot.policy.pt`

Points/hand is from the bot's perspective; seats are swapped at the halfway point of every match.

| Opponent | Hands | Points/hand (95% CI) | Decisive | Win rate | Gin | Knock | Undercut | Dead | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smart-draw-rule | 2000 | +2.30 ± 1.10 | 100% | 36.2% | 8.5% | 61.9% | 29.5% | 0.1% | **bot wins** |
| novice-rule | 2000 | +16.23 ± 1.12 | 98% | 62.3% | 19.9% | 35.4% | 42.3% | 2.5% | **bot wins** |
| random | 2000 | +76.55 ± 1.11 | 95% | 99.2% | 94.0% | 0.4% | 0.2% | 5.3% | **bot wins** |
| strong-teacher | 2000 | -1.37 ± 1.37 | 95% | 48.4% | 95.3% | 0.0% | 0.0% | 4.7% | opponent wins |
| results/final/gin_rummy_bot.policy.pt | 2000 | +7.27 ± 1.14 | 100% | 46.5% | 16.9% | 49.0% | 33.8% | 0.2% | **bot wins** |

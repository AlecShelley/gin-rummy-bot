# Benchmark: `results/v2_final/gin_rummy_bot.policy.pt`

Points/hand is from the bot's perspective; seats are swapped at the halfway point of every match.

| Opponent | Hands | Points/hand (95% CI) | Decisive | Win rate | Gin | Knock | Undercut | Dead | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smart-draw-rule | 2000 | +1.89 ± 1.11 | 100% | 36.1% | 8.1% | 62.0% | 29.9% | 0.0% | **bot wins** |
| novice-rule | 2000 | +14.69 ± 1.13 | 98% | 57.9% | 19.7% | 39.6% | 38.4% | 2.5% | **bot wins** |
| random | 2000 | +74.48 ± 1.21 | 92% | 99.3% | 91.2% | 0.5% | 0.5% | 7.8% | **bot wins** |

# Benchmark: `results/final/gin_rummy_bot.policy.pt`

Points/hand is from the bot's perspective; seats are swapped at the halfway point of every match.

| Opponent | Hands | Points/hand (95% CI) | Decisive | Win rate | Gin | Knock | Undercut | Dead | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| novice-rule | 1500 | +8.37 ± 1.14 | 100% | 63.8% | 3.9% | 92.8% | 3.1% | 0.1% | **bot wins** |
| random | 1500 | +52.74 ± 0.87 | 100% | 99.5% | 4.1% | 95.8% | 0.1% | 0.0% | **bot wins** |
| results/pretrained.pth | 1500 | +3.34 ± 1.20 | 100% | 60.9% | 8.7% | 85.1% | 6.0% | 0.2% | **bot wins** |

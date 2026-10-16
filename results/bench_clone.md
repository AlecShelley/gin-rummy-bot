# Benchmark: `results/pretrained.pth`

Points/hand is from the bot's perspective; seats are swapped at the halfway point of every match.

| Opponent | Hands | Points/hand (95% CI) | Decisive | Win rate | Gin | Knock | Undercut | Dead | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random | 800 | +44.11 ± 2.03 | 80% | 98.4% | 8.4% | 71.5% | 0.0% | 20.1% | **bot wins** |
| novice-rule | 800 | -9.02 ± 1.52 | 97% | 28.8% | 2.8% | 91.5% | 3.1% | 2.6% | opponent wins |
| results/dagger/best.policy.pt | 800 | -3.69 ± 1.55 | 92% | 41.3% | 3.9% | 84.4% | 4.0% | 7.8% | opponent wins |

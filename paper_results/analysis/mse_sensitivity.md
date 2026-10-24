# Near-duplicate MSE threshold sensitivity

Candidates (pHash Hamming <= 2, eval splits): **129**
Eval images actually removed at MSE <= 900: **30**
Largest MSE among removed leaks: **817.85**
Smallest MSE among surviving candidates: **958.6**
Margin (nearest survivor / 900): **1.065x**

| MSE <= | flagged | true leaks captured | survivors flagged |
|---:|---:|---:|---:|
| 250 | 18 | 18 | 0 |
| 500 | 25 | 25 | 0 |
| 750 | 29 | 29 | 0 |
| 900 | 30 | 30 | 0 |
| 1100 | 38 | 30 | 8 |
| 1300 | 51 | 30 | 21 |
| 1500 | 65 | 30 | 35 |
| 2000 | 95 | 30 | 65 |

# MLMET_Reproduce
USC CSCI 662 Project

## Results

### Table 2: Comparison of Patterns

Pattern | Precision | Recall | F1 
--- | --- | --- | --- |
M and any other H | 21.8 | 33.1 | 26.3
M and some other H | 19.8 | 34.4 | 25.1
H such as M | 21.6 | 23.0 | 22.3
such H as M | 14.8 | 25.1 | 18.6
H including M | 22.4 | 15.0 | 18.0
H especially M | 20.1 | 5.1 | 8.2

* Note that instead of selecting top 10 valid predictions for each entity, we select valid predictions from top 10 predictions. This setting results in higher F1 scores for top 5 patterns.

### Table 4: Comparison of Weakly Labeled Data

Method | Precision | Recall | F1 
--- | --- | --- | --- |
BERT-Ultra-Direct | 52.3 | 31.9 | 39.6
BERT-Ultra-Pre |  |  |  |
Ours (Single Pattern, Unweighted Loss, No self-train) |  |  |  |




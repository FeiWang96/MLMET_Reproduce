# MLMET_Reproduce
USC CSCI 662 Project

## Claim Verification

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
BERT-Ultra-Pre | 52.1 | 33.6 | 40.8 
Ours (Single Pattern) | 52.1 | 36.7 | 43.0 

* None of the above uses Weighted Loss or self-train, so the performance may be slightly worse than the numbers reported in the paper.
* For weak supervision, we only use 300,000 instances.


## Additional Ablations

### Influence of K

K | Precision | Recall | F1 
--- | --- | --- | --- |
5 | 28.6 | 23.2 | 25.6
10 | 21.8 | 33.1 | 26.3
15 | 17.7 | 39.0 | 24.3 
20 | 15.0 | 43.0 | 22.3  


* We use 'M and any other H'.

### Input Format

Entity Marker | Precision | Recall | F1 
--- | --- | --- | --- |
Repeat at End | 52.3 | 31.9 | 39.6
Surround by Special Token | 52.6 | 31.5 | 39.4  

* We use BERT-Ultra-Direct model.
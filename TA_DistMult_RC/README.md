# TA-DistMult-RC
This code is modified based on https://github.com/BorealisAI/DE-SimplE


## Reproducing results

In order to reproduce the results on the datasets in the paper, run the following commands
```
python main.py --dataset icews14 -dropout 0.4 -se_prop 1.0 -model TA_DistMult_SED

python main.py --dataset icews14 -dropout 0.4 -se_prop 1.0 -model TA_DistMult_MLP
```






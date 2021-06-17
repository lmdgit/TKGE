# TNTComplEx_RC
This code is modified based on https://github.com/facebookresearch/tkbc

## Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name tkbc_env python=3.7
source activate tkbc_env
conda install --file requirements.txt -c pytorch
```

Then install the kbc package to this environment
```
python setup.py install
```

## Datasets

To download the datasets, go to the tkbc/scripts folder and run:
```
chmod +x download_data.sh
./download_data.sh
```

Once the datasets are downloaded, add them to the package data folder by running :
```
python tkbc/process_icews.py
```

This will create the files required to compute the filtered metrics.

## Reproducing results

In order to reproduce the results on the smaller datasets in the paper, run the following commands

```
python tkbc/learner.py --dataset ICEWS14 --model TNTComplEx_SED --rank 156 --emb_reg 1e-2 --time_reg 1e-2 --rc_type 0

python tkbc/learner.py --dataset ICEWS14 --model TNTComplEx_MLP --rank 156 --emb_reg 1e-2 --time_reg 1e-2 --rc_type 1
```




## License
tkbc is CC-BY-NC licensed, as found in the LICENSE file.

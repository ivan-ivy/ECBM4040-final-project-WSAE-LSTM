# ECBM4040-final-project-WSAE-LSTM
### Author: Yifan Liu yl4314

This is the final project of course ECBM 4040 Neural Networks and Deep Learning at Columbia University. The project is aim to replicate the WSAE-LSTM model from the paper [A deep learning framework for financial time series using stacked autoencoders and long- short term memory](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180944).  

The data set used in this project is provided by the original paper and can be found [here](https://figshare.com/s/acdfb4918c0695405e33).  


## Structure 
The project structure follows the guidance of [cookiecutter-data-science](http://drivendata.github.io/cookiecutter-data-science/#directory-structure)


```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Normalized data
│   ├── processed      <- Data after WT and SAEs transformed.
│   └── raw            <- The original data.
│
│
├── models             <- Trained and serialized LSTM models
│
├── notebooks          <- Notebooks for training and evaluation
│   │     
|   ├── training.ipynb <- for training or reproducing result
|   └── evaluation.ipynb    <- used to show results
|
├── references         <- Reference papers.
│
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models。
│   │   │            
│   │   ├── LSTM.py      <- LSTM models
│   │   ├── stacked_auto_encoder.py  <- SAEs models
│   │   ├── metrics.py   <- custom metrics for evaluation    
│   │   └── wavelet.py   <- wavelet transform       
│   │
│   └── visualization  <- Scripts to show final result
│       └── visualize.py
│
└── .gitignore   <- preclude unnecessary files in git           
```

## Instruction of running code
The trained model are stored in models directory, using `training.ipynb` under `./notebook` directory can use the trained model to make prediction and reproduce the results. 
To see the final evaluation, run `evaluation.ipynb` under `./notebook` directory.


## Key functions
Key models of this project are included in `./src/models` directory, including wavelet tranform (WT), stacked autoencoder (SAEs), and LSTM.
The WT an SAEs are used in `generate_feature.py` under `./src/feature` directory. And LSTM models are called in `training.ipynb` notebook.
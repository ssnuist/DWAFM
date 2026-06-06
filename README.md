# DWAFM

The letter “DWAFM:Dynamic Weighted Graph Structure Embedding Integrated with Attention and Frequency-Domain MLPs for Traffic Forecasting”，which has been submitted to *IEEE Signal Processing Letters (SPL)* and is currently under review.  

## Datasets

The following table summarizes the statistical characteristics of the five real-world traffic datasets evaluated in our experiments:

| Datasets | Nodes | Length | Sample Rate | Type |
| :--- | :---: | :---: | :---: | :--- |
| **PEMS08** | 170 | 17856 | 5 min | Traffic flow |
| **PEMS04** | 307 | 16992 | 5 min | Traffic flow |
| **PEMS03** | 358 | 26208 | 5 min | Traffic flow |
| **PEMSD7(M)** | 228 | 12672 | 5 min | Traffic speed |
| **PEMSD7(L)** | 1026 | 12672 | 5 min | Traffic speed |

## Reproducibility (all baseline models)

### Environment
- python 3.11 
- torch 2.3.1  + cu121
- numpy 1.26.4

To ensure reproducibility, we fix random seeds and configure CuDNN as follows:

```python
CFG.ENV.SEED = 1
CFG.ENV.DETERMINISTIC = True
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True
CFG.ENV.CUDNN.BENCHMARK = False
CFG.ENV.CUDNN.DETERMINISTIC = True
```
## Acknowledgement
Our work is developed based on BasicTS, and all baseline models are derived from: [https://github.com/GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS)

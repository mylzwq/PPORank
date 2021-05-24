## PPORank
Deep reinforcement learning for personalized drug recommendation
## Dependecies
**Python**>=3.7.10; torch(1.18.1); yaml(0.1.7);
numpy(>=1.18.1); pandas(1.0.0); scipy(1.4.1); seaborn(0.8.1); tensorboard (1.15.5);
matplotlib(3.1.1); joblib(0.14.1); json5(0.8.5);jupyter(1.0.0); tqdm (4.42.0)
For drugs processing: deepchem (2.5.0); h5py(2.10.0); hdf5(1.10.4)

## PPORank usage
The implementation of PPORank is in "main.py":
(cpu version) :
The following is an exmaple for GDSC dataset when training with 16 actors, projection dimension of 100 without normalize y 
more details can be found in arguments.py

```
python main.py   --num_processes 16  --nlayers_deep 2 --Data GDSC_ALL --analysis FULL  --algo ppo --f 100  --normalize_y 
```
model logs dir : ./logs

model saved dir: ./Saved

model prediction saved dir: ./results

The clean data could be found in 

[Data Sharing](https://drive.google.com/drive/folders/1-YcEcRP6IObhT8ojes9L29Z54P-japjJ?usp=sharing)

## Scripts for runing the experiments
Download and preprocess the GDSC dataset:
```
python ./preprocess/load_dataset.py load_GDSC.txt
```
Split the data for training and testing, create folds for cross-validation and Pretrain the MF layers' weight
```
python prepare.py config.yaml

```
Runing the experiments on ppo (with config file "./configs/configS_base.yaml"):

```
python results.py > results_ppo.txt

```
PPO experiment with TCGA cohort

```
 python load_TCGA.py
 
 python results_TCGA.py ./TCGA/TCGA_BRCA.npz 
```




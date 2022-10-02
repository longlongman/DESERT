# DESERT
Zero-Shot 3D Drug Design by Sketching and Generating (NeurIPS 2022)

<!-- ![](./pics/sketch_and_generate.png)
![](./pics/overview.png) -->
<div  align="center">    
<img src="./pics/sketch_and_generate.png"/>
</div>
<div  align="center">    
<img src="./pics/overview.png"/>
</div>

P.s. Because the project is too tied to ByteDance infrastructure, we can not sure that it can run on your device painlessly.

## Requirement
Our method is powered by an old version of [ParaGen](https://github.com/bytedance/ParaGen) (previous name ByCha).

Install it with
```bash
cd mybycha
pip install -e .
pip install horovod
pip install lightseq
```
You also need to install
```bash
conda install -c "conda-forge/label/cf202003" openbabel # recommend using anaconda for this project 
pip install rdkit-pypi
pip install pybel scikit-image pebble meeko==0.1.dev1 vina pytransform3d
```

## Pre-training

### Data Preparation
Our training data was extracted from the open molecule database [ZINC](https://zinc.docking.org/). You need to download it first. 

To get the fragment vocabulary
```bash
cd preparation
python get_fragment_vocab.py # fill blank paths in the file first
```

To get the training data
```bash
python get_training_data.py # fill blank paths in the file first
```

We also provide partial training data and vocabulary [Here](https://drive.google.com/drive/folders/1T2tKgILJAIMK6uTuhh3-qV-Ib0JVgaBs?usp=sharing).

### Training Shape2Mol Model

You need to fill blank paths in configs/training.yaml and train.sh.

```bash
bash train.sh
```

We also provide a trained checkpoint [Here](https://drive.google.com/file/d/1YCRORU5aMJEMO8hDT_o9uKCXmXTL5_5N/view?usp=sharing).

## Design Molecules

### Sketching

For a given protein, you need to get its pocket by using [CAVITY](http://www.pkumdl.cn:8000/cavityplus/computation.php).

Sampling molecular shapes with
```bash
cd sketch
python sketching.py # fill blank paths in the file first
```

### Generating

```bash
bash generate.sh # fill blank paths in the file first
```

## Citation
```
@inproceedings{long2022DESERT,
  title={Zero-Shot 3D Drug Design by Sketching and Generating},
  author={Long, Siyu and Zhou, Yi and Dai, Xinyu and Zhou, Hao},
  booktitle={NeurIPS},
  year={2022}
}
```

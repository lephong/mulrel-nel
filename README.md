mulrel-nel: Multi-relational Named Entity Linking
========

A Python implementation of Multi-relatonal Named Entity Linking described in 

[1] Phong Le and Ivan Titov (2018). [Improving Entity Linking by 
Modeling Latent Relations between Mentions](https://arxiv.org/pdf/1804.10637.pdf). ACL 2018.

Written and maintained by Phong Le (ple [at] inf.ed.ac.uk)


### Installation

- Requirements: Python 3.5 or 3.6, Pytorch 0.3, CUDA 7.5 or 8

### Usage

The following instruction is for replicating the experiments reported in [1]. 


#### Data

Download data from [here](https://drive.google.com/open?id=1IDjXFnNnHf__MO5j_onw4YwR97oS8lAy) 
and unzip to the main folder.

#### Train

To train a 3-relation ment-norm model, from the main folder run 

    export PYTHONPATH=$PYTHONPATH:../
    python -u -m nel.main --mode train --n_rels 3 --mulrel_type ment-norm --model_path model
 
Using a GTX 1080 Ti GPU it will take about 1 hour. The output is a model saved in two files: 
`model.config` and `model.state_dict` . 

#### Evaluation

Execute

    python -u -m nel.main --mode eval --model_path model


# Class Incremental Domain Adaptation for MTL-SD based Surgical Scene Understanding

<!---------------------------------------------------------------------------------------------------------------->


<!---------------------------------------------------------------------------------------------------------------->
## Feature Extractor
To be added
<!---------------------------------------------------------------------------------------------------------------->
## Scene graph
<!---------------------------------------------------------------------------------------------------------------->
To be added
<!---------------------------------------------------------------------------------------------------------------->

## Caption
<!---------------------------------------------------------------------------------------------------------------->
To be added
<!---------------------------------------------------------------------------------------------------------------->

## Code Overview
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch and DGL library and there are three main folders: 

- `data/`: .
- `datasets/`: Contains the dataset needed to train the network.
- `checkpoints/`: Conatins trained weights
- `evaluation/`: Contains utility tools for evaluation.
- `Feature_extractor/`: Used to extract features from dataset images to train the scene graph and caption network for algorithm 1, stage 3 & 4.
- `models/`: Contains network models.
- `utils/`: Contains utility tools used for training and evaluation.

<!---------------------------------------------------------------------------------------------------------------->

## Library Prerequisities.

### DGL
<a href='https://docs.dgl.ai/en/latest/install/index.html'>DGL</a> is a Python package dedicated to deep learning on graphs, built atop existing tensor DL frameworks (e.g. Pytorch, MXNet) and simplifying the implementation of graph-based neural networks.

### Prerequisites
- Python 3.6
- Pytorch 1.1.0
- DGL 0.3
- CUDA 10.0
- Ubuntu 16.04


### Dataset
#### Download feature extracted data for training and evalutation
1. endovis challange 2018
2. Download the pretrain word2vec model on [GoogleNews](https://code.google.com/archive/p/word2vec/) and put it into `datasets/word2vec` 

### Training
- Algorithm 1:
  - scene_graph_train.py (stage 3)
  - Caption_train.py (stage 4)
  - MTL_train_finetune.py (stage 5)
- Algorithm 2:
  - MTL_KD_train.py
- Checkpoints will be saved in `checkpoints/` folder.

### Evaluation
- MTL_evaluation.py


### Acknowledgement
Code adopted and modified from :
1. Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion
    - Paper [Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion](https://arxiv.org/abs/2001.02302).
    - Official Pytorch implementation [code](https://github.com/birlrobotics/vs-gats).
2. End-to-End Incremental Learning
    - Paper [End-to-End Incremental Learning](ttps://arxiv.org/pdf/1807.09536.pdf).
    - Pytorch implementation [code](https://github.com/fmcp/EndToEndIncrementalLearning).
3. Curriculum by smoothing
    - Paper [Curriculum by smoothing](https://arxiv.org/pdf/2003.01367.pdf).
    - Pytorch implementation [code](https://github.com/pairlab/CBS).
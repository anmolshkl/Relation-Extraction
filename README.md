
# Overview

In this project, I have implemented bi-directional GRU with attention as well as an original model for Relation Extraction.

The GRU is loosely based on the approach done in the work of Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification (Zhou et. al, 2016).

Additionally, I have also implemented my own network architecture to solve this task. More details are given in the report.pdf.


# Installation

This project is implemented in python 3.6 and tensorflow 2.0. Follow these steps to setup the environment:

1. [Download and install Conda](http://https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")
2. Create a Conda environment with Python 3.6

```
conda create -n nlp-hw4 python=3.6
```

3. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code.
```
conda activate nlp-hw4
```
4. Install the requirements:
```
pip install -r requirements.txt
```

5. Download spacy model
```
python -m spacy download en_core_web_sm
```

6. Download glove wordvectors:
```
./download_glove.sh
```

# Data

The training and validation data is in the form of text files. Training can be found in `data/train.txt` and validation is in `data/val.txt`. We are using data from a previous SemEval shared task which in total had 8,000 training examples. The train/validation examples are a 90/10 split from this original 8,000. More details of the data can be found in the overview paper SemEval-2010 task 8: multi-way classification of semantic relations between pairs of nominals (Hendrickx et. al, 2009) as well as extra PDFs explaining the details of each relation in the dataset directory.


# Code Overview

## Train and Predict

There are 4 main scripts in the repository `train_basic.py`, `train_advanced.py`, `predict.py` and `evaluate.pl`.

- Train scripts do as described and saves the model to be used later for prediction. Basic training script trains the basic `MyBasicAttentiveBiGRU` model (Bi-RNN+attention).

- Predict generates predictions on the test set `test.txt` and saves the output to a file.

- Evaluation script is the pearl script unlike others. One can use it see detailed report of the predictions file against the gold labels.


#### Train a model
```
python train_basic.py --embed-file data/glove.6B.100d.txt --embed-dim 100 --batch-size 10 --epochs 5
# Remove the --data-file data/train_fixture.txt to run on the full training set
python train_advanced.py --embed-file data/glove.6B.100d.txt --embed-dim 100 --batch-size 10 --epochs 5 --data-file data/train_fixture.txt
# stores the model by default at : serialization_dirs/basic/
```

#### Predict with model
```
python predict.py --prediction-file my_predictions.txt --batch-size 10 --load-serialization-dir serialization_dirs/advanced/
```

## Extra Scripts

The `download_glove.sh` downloads all the GloVe embeddings of all the dimensions.


## Experimentations

Different experiments performed as part of the projects were as follows - 

1. Run with only Word Embeddings (remove `pos_inputs` and dependency structure. Removing dep structure can be done by setting `shortest_path = []` in `data.py`)
2. Run with only Word + Pos embeddings
3. Run with only Word + Dep structure


## Advanced Model

The advanced model can be found in model.py (MyAdvancedModel). It is based on Convolution Neural Network and the detailed information regarding the implementation can be found in report.pdf and the code.
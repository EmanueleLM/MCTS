# Monte Carlo Tree Search for Multiple Words Substitutions
This code is the original implementation to replicate the Monte Carlo Tree Search (MCTS) simulations as described in the paper "Assessing Robustness of Text Classification through Maximal Safe Radius Computation", accepted as long paper publication at EMNLP-Findings (EMNLP2020).
<br/>
ArXiv: https://arxiv.org/abs/2010.02004

## Dependencies (versions reported have been tested)
You need to install python3, then you have to install the following dependencies (e.g., by pip3): we list the packages versions that we used, but other versions could work:
- Numpy 1.17.2
- Nltk 3.4.5
- Pandas 0.25.1
- Progress 1.5
- Tensorflow 1.14.0
- Torch 1.3.1
- Torchtext 0.4.0

Further to this, you need to download the GloVe and GloVeTwitter pre-trained embeddings respectively from http://nlp.stanford.edu/data/glove.6B.zip and http://nlp.stanford.edu/data/glove.twitter.27B.zip. Once archives have been unzipped, move them inside the ```./data/embeddings``` folder. Please note that naming should be consistent with variables in `ub_CNN_models.py` and `ub_LSTM_models.py` files, i.e., `glove.6B.<DIMS>d.txt` and `glove.twitter.27B.<DIMS>d.txt` where <DIMS> is the dimensionality of the embedding (can be specified as a parameter, see the next section). For example, to test GloVe50d, the embedding's name should be `glove.6B.50d.txt`.

## Instructions
To test all the experiments for CNNs, launch from console:
<br/>
```python3 ub_CNN_models.py```
<br/>
While for LSTM models, run:
<br/>
```python3 ub_LSTM_models.py```
<br/>

Please note that you can pass several arguments to these scripts:
<br/>
```-s, --sims: number of Monte Carlo simulations per-vertex```
<br/>
```-m, --max-depth: maximum depth of the tree (i..e, number of perturbations)```
<br/>
```-e, --eps: max-distance (in L2 norm) for collecting neighbors```
<br/>
```-l, --lrate: UTC learning rate```
<br/>
```-d, --discount: UTC discount factor```
<br/>
```-ed, --emb-dims: embeddings dimension```
<br/>
```-hu, --lstm-hu: LSTM hidden units (applicable only to `ub_LSTM_models.py`)```
<br/>

The first time it may take a while as MCTS collects all the neighbors for every word in an input text and it saves them in a file named 'neighbors_<TEST_NUMBER>.pkl' in 'obj' folder (<TEST_NUMBER> is the the i-th input text from test set). 

# Cite
If you are considering citing this algorithm as part of your work, please use the following bibtex:
<br/>
```
@article{la2020assessing,
  title={Assessing Robustness of Text Classification through Maximal Safe Radius Computation},
  author={La Malfa, Emanuele and Wu, Min and Laurenti, Luca and Wang, Benjie and Hartshorn, Anthony and Kwiatkowska, Marta},
  journal={arXiv preprint arXiv:2010.02004},
  year={2020}
}
```


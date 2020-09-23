# Monte Carlo Tree Search for Multiple Words Substitutions
This code is the original implementation to replicate the Monte Carlo Tree Search (MCTS) simulations as described in the paper "Assessing Robustness of Text Classification through Maximal Safe Radius Computation", accepted as long paper publication at EMNLP-Findings (EMNLP2020).
ArXiv: <link>

## Dependencies (versions reported have been tested)
You need to install python3, then you have to install the following dependencies (e.g., by pip3): we list the packages versions that we used, but other versions could work:
- Tensorflow==2.0.0
- Numpy==1.17.2
- Progress==1.5
- Torch==1.3.1

Further to this, you need to download the GloVe and GloVeTwitter pre-trained embeddings respectively from http://nlp.stanford.edu/data/glove.6B.zip and http://nlp.stanford.edu/data/glove.twitter.27B.zip. Once archives have been unzipped, move them inside the ```./data/embeddings``` folder. Please note that naming should be consistent with variables in `ub_CNN_models.py` and `ub_LSTM_models.py` files, i.e., `glove.6B.<DIMS>d.txt` and `glove.twitter.27B.<DIMS>d.txt` where <DIMS> is the dimensionality of the embedding (can be specified as a parameter, see the next section). For example, in roder to test GloVe50d, the embedding's name should be `glove.6B.50d.txt` and so on.

## Instructions
Before testing the code, you just need to download the embedding, which by default is 'glove.twitter.27B.25d.txt', that can be downloaded <a href="http://nlp.stanford.edu/data/glove.twitter.27B.zip">here</a>, extract 'glove.twitter.27B.25d.txt' (you can possibly delete all the others if you won't use them) and put it into the folder 'data'. The name of the embedding is 
'glove.twitter.27B.25d.txt', so if you change it, also change 'lowerbound_mcts_imdb.py' to be consistent.

To test all the experiments for CNNs, launch from console:
<br/>
```
python3 ub_CNN_models.py
```
<br/>
While for LSTM models, run:
<br/>
```
python3 ub_LSTM_models.py
```
<br/>

Please note that you can pass arguments to these scripts:
<br/>
```
-s, --sims: number of Monte Carlo simulations per-vertex
```
<br/>
```
-m, --max-depth: maximum depth of the tree (i..e, number of perturbations)
```
<br/>
```
-e, --eps: max-distance (in L2 norm) for collecting neighbors
```
<br/>
```
-l, --lrate: UTC learning rate
```
<br/>
```
-d, --discount: UTC discount factor
```
<br/>
```
-ed, --emb-dims: embeddings dimension
```
<br/>
```
-hu, --lstm-hu: LSTM hidden units (applicable only to `ub_LSTM_models.py`)
```
<br/>

The first time it may take a while as MCTS collects all the neighbors for every word in an input text and it saves them in a file named 'neighbors_<TEST_NUMBER>.pkl' in 'obj' folder (<TEST_NUMBER> is the the i-th input text from test set). 

# Cite
If you are considering citing this work, please use the following bibtex snippet:
<INSERT-PAPER-BIBTEX>
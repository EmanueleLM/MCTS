"""
MCTS Upper bound for LSTM models.
It is used as upper bound technique for models verified with POPQORN.
"""
import argparse
import numpy as np
import pickle
import time
import torch
import sys
from os import path
from progress.bar import Bar

import MCTS
from load_data import load_NEWS_dataset
from MCTS_override import __override_simulate, __override_simulate_single_node_torch
from Neighbors import Neighbors

# Parse epsilon and window size from command line
parser = argparse.ArgumentParser(description='Number of simulations, max-depth of the tree and max-distance (in L2 norm) for collecting neighbors can be passed as arguments.')
parser.add_argument('-s', '--sims', dest='sims', type=int, default=500, help='number of Monte Carlo simulations per-vertex')
parser.add_argument('-m', '--max-depth', dest='max_depth', type=int, default=5, help='maximum depth of the tree (i..e, number of perturbations)')
parser.add_argument('-e', '--eps', dest='eps', type=float, default=10.0, help='max-distance (in L2 norm) for collecting neighbors')
parser.add_argument('-l', '--lrate', dest='l_rate', type=float, default=1e-3, help='UTC learning rate')
parser.add_argument('-d', '--discount', dest='discount', type=float, default=0.5, help='UTC discount factor')
parser.add_argument('-ed', '--emb-dims', dest='embedding_dims', type=int, default=100, help='embeddings dimension')
parser.add_argument('-hu', '--lstm-hu', dest='lstm_hu', type=int, default=256, help='LSTM hidden units')
args = parser.parse_args()
# assign parsed parameters
n_sims = args.n_sims
max_depth = args.max_depth
eps = args.eps
l_rate = args.l_rate
discount = args.discount
embedding_dims = args.embedding_dims
lstm_hu = args.lstm_hu

# Global parameters
path_to_embeddings = './data/embeddings/'
path_to_models = './models/lstm_hu-{}/'.format(lstm_hu)
DATASETS = ['news']  # 'news' dataset implemented so far
MAXLEN = [14]
EMBEDDING_PREFIX = ['glove', 'glovetwitter']
EMBEDDING_TORCHNAME = ['glove.6B.{}d'.format(embedding_dims), 'glove.twitter.27B.{}d'.format(embedding_dims)]
EMBEDDING_PATH = [path_to_embeddings + 'glove.6B.{}d.txt'.format(embedding_dims),
                  path_to_embeddings + 'glove.twitter.27B.{}d.txt'.format(embedding_dims)]
MODEL_PATH = [path_to_models + 'lstm-glove{}d'.format(embedding_dims),
              path_to_models + 'lstm-glovetwitter{}d'.format(embedding_dims)]

for (dataset, maxlen) in zip(DATASETS, MAXLEN):
    for (embedding_prefix, embedding_path, embedding_torchname, model_path) in zip(EMBEDDING_PREFIX, EMBEDDING_PATH, EMBEDDING_TORCHNAME, MODEL_PATH):
        # Start measuring time
        start_time = time.time()  

        # 1. load dataset and neural network (Torch is strange)
        print("[logger]: Loading {} dataset with maxlen={}, emb_dims={}".format(dataset, maxlen, embedding_dims))
        k = min(1000, n_sims)  # this should be larger than the number of experiments
        if dataset == 'news':
            (_,_), (X_test, y_test), model = load_NEWS_dataset(embedding_prefix, embedding_dims, maxlen, num_samples=-1, return_text=False)
            (_,_), (x_text, x_indices) = load_NEWS_dataset(embedding_prefix, embedding_dims, maxlen, num_samples=-1, return_text=True)
        
        # 2. load model's weights
        W = torch.load(model_path)
        E = W['embedding.weight']
        W.pop('embedding.weight', None)  # drop embedding as it has been moved outside
        model.load_state_dict(W)
        #  check accuracy on this set
        correct = 0
        for x,y in zip(X_test, y_test): 
            if np.argmax(model.predict(x)) == np.argmax(y): 
                correct += 1 
        print("[logger]: Accuracy of the model on {} samples is {}".format(len(X_test), correct/len(X_test)))
        
        # 3. select the test point and accumulate the neighbors of each word (this will speedup MCTS a lot)
        n = Neighbors(embedding_path,
                    in_memory=False,
                    norm=np.linalg.norm,
                    neighbors_filters=[])
        # Statistics of the attacks
        total_perturbations, total_attacks, number_verified = 0, 0, 0
        resume_perturbations = []
        for i in range(len(x_text)):
            test_pt = i
            x, y = (X_test[test_pt], y_test[test_pt])
            input_text = x_text[test_pt][:maxlen]
            neighbors_prefix_path = "./obj/{}_test_set/{}/{}d/".format(dataset, embedding_prefix, embedding_dims)
            if path.exists(neighbors_prefix_path + "neighbors_{}.pkl".format(test_pt)):
                print("[logger]: Loading neighbors for each word from saved file.")
                nearest_neighbors = pickle.load(open(neighbors_prefix_path + "neighbors_{}.pkl".format(test_pt), 'rb'))
            else:
                nearest_neighbors = {}
                print("[logger]: Gathering neighbors for each word (this may take lot of time).")
                bar = Bar('[logger]: Processing', max=maxlen)
                for word in input_text:
                    nearest_neighbors[word] = [item[0] for item in n.nearest_neighbors(word, eps, k)]
                    bar.next()
                bar.finish()
                print("[logger]: Saving dictionary of neighbors to speedup next time.")
                with open(neighbors_prefix_path + "neighbors_{}.pkl".format(test_pt), 'wb') as f:
                    pickle.dump(nearest_neighbors, f, pickle.HIGHEST_PROTOCOL)

            # 4. create the MCTree and instantiate the search
            branch_factor = len(input_text)
            MCTS.MCTree.__simulate_single_node = __override_simulate_single_node_torch  # override MCTS __simulate_single_node method
            MCTS.MCTree.simulate = __override_simulate  # override MCTS simulate method
            tree = MCTS.MCTree(branch_factor, max_depth, n_sims, l_rate, discount)
            if np.argmax(model.predict(x)) != np.argmax(y):
                print("[logger-ERROR]: Prediction and true label are different: can't proceed in the analysis.")
                continue
            else:
                number_verified += 1
            y_hat = np.max(model.predict(x))  # this is used for the 'gain' and hence for the MCTS-UCT heuristc
            true_label = np.argmax(model.predict(x))

            while tree.actual_depth != tree.max_depth:
                v_star = tree.select()
                print("Node selected {} (depth={}), whose parent is {}".format(v_star, v_star.depth, v_star.parent))
                assert v_star in tree.T, "Error: node selected is not terminal."
                tree.expand(v_star)
                _, nnn, mean_, std_ = tree.simulate(v_star, model.predict, (nearest_neighbors, x, y_hat, true_label, input_text, k, n.word2index, n.index2word, n.index2embedding, 1), False)

                total_perturbations += (1 if nnn>0 else 0)
                total_attacks += nnn

                if ~np.isnan(mean_):
                    resume_perturbations.append([i, mean_, std_])

                tree.backpropagate(v_star)

            # 5. print statistics on attacks
            print("[logger]: Text {}".format(i))
            print("\t  Number of pertubed words {}/{}".format(nnn, branch_factor))

        # 6. log results
        # print
        print("[logger]: LSTM-{}-{}d\n".format(dataset, embedding_dims))
        print("[logger]: Embedding {}".format(embedding_prefix))
        print("[logger]: #Success/#Texts={}/{}".format(total_perturbations, number_verified+1))
        print("[logger]: Number of Perturbations(over {} texts)={}".format(number_verified+1, total_attacks))
        print("[logger]: Indices, Means and Stds of each Successfull Perturbation\n")
        print("[logger]: {}\n".format(resume_perturbations))
        print("[logger]: Exec Time {}".format(time.time() - start_time))
        # save logs
        log_results = "LSTM-{}-{}d\n".format(dataset, embedding_dims)
        log_results += "[logger]: Embedding {}\n".format(embedding_prefix)
        log_results += "[logger]: #Success/#Texts={}/{}\n".format(total_perturbations, number_verified+1)
        log_results += "[logger]: Number Perturbations(over {} texts)={}\n".format(number_verified+1, total_attacks)
        log_results += "[logger]: Exec Time {}\n".format(time.time() - start_time)
        log_results += "[logger]: Indices, Means and Stds of each Successfull Perturbation\n"
        log_results += "{}\n".format(resume_perturbations)
        f = open("./results/results-{}50d".format(embedding_prefix), "a")
        f.write(log_results)
        f.close()

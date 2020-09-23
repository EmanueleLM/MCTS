import copy as cp
import numpy as np
import torch

def __override_simulate(self, v, sim, args, simd=False):
    """
    Execute n_sims simulations wiht function sim of every child of node v
    TODO: implement SIMD support for simulation of batches.
    """
    num_attacks = 0
    num_perturbations = 0
    best_norms = []
    for c in v.children:
        if simd is False:
            e, n, b = self.__simulate_single_node(c, sim, args)  
        else:
            raise Exception("Not Implemented Exception.")
        if e is True:
            num_attacks += 1
            num_perturbations += n
            best_norms.append(b)
    # nan appears as mean/std for unsuccessful perturbations
    if len(best_norms) == 0:
        mean_ = std_ = np.nan
    else:
        mean_, std_ = np.asarray(best_norms).mean(), np.asarray(best_norms).std()
    return num_attacks, num_perturbations, mean_, std_
    

def __override_simulate_single_node(self, v, sim, args):
    """
    This is used to override the class 'MCTree.__simulate_single_node'
    Execute n_sims of simulation from node v: sim function is executed
        taking any argument (in args) and returning a floating point which is the update
        dQ to backpropagate.
    Ideally args will have some reference to a variable that is unique of the vertex,
        hence you can put into args references to the vertex index, for example. 
    """
    #print("[logger]: Processing node {} at depth {}".format(v.index, v.depth))
    neighbors, x, y, true_label, text, n_size, word2index, index2word, index2embedding, normalization = args[0], cp.copy(args[1]), args[2], args[3], cp.copy(args[4]), args[5], args[6], args[7], args[8], args[9]
    if v.children != None:
        raise Exception("Expanding Exception: node at depth {} with index {} has children hence it can't be expanded".format(v.depth, v.index))
    # extract the series of perturbations to sample
    perturbation_indices = []
    tmp = v
    while tmp.parent != None:
        perturbation_indices.append(tmp.index)
        # uncomment this if you want a fast proxy to to try multiple perturbations without exploring the tree
        p_1 = np.random.randint(0, len(text))
        perturbation_indices.append(p_1)
        tmp = tmp.parent
    # compute simulation indices
    permutations = np.random.randint([0 for _ in range(len(perturbation_indices))],
                                     [len(neighbors[text[v]]) for v in perturbation_indices], 
                                     [self.n_sims, len(perturbation_indices)])
    # create a single vector that exploit SIMD forward of the neural network
    X = np.tile(x, (self.n_sims, 1, 1, 1)).reshape(self.n_sims, x.shape[1]*x.shape[2], x.shape[3])
    for i in range(self.n_sims):
        for (j,n) in zip(perturbation_indices, range(len(perturbation_indices))):
            #print("Sim {} , mutating word {} at index {} with {}".format(i, text[j], j ,permutations[i,n]))
            X[i,j] = index2embedding[word2index[neighbors[text[j]][permutations[i,n]]]]/normalization  # normalize to be consistent
    #print("[logger] At index {} max perturbation leaded to {} to class 0".format(v.index, np.min(sim(X)[:,true_label])))
    X = X.reshape(self.n_sims, x.shape[1], x.shape[2], x.shape[3])

    effective, num_perturbations = False, 0
    worst_accuracy_drop = np.min(sim(X)[:,true_label,np.newaxis] - sim(X))
    l2norm_best_attack = 0.
    if worst_accuracy_drop < 0:
        # report all label-changing words of length 1
        single_perturbations = []
        if v.depth == 1:
            for i in range(len(permutations)):
                a = sim(X[i].reshape(1, x.shape[1], x.shape[2], x.shape[3]))
                if a[0,true_label] - np.max(a) < 0.:
                    single_perturbations.append(neighbors[text[v.index]][permutations[i][0]])        
        ii = np.argmin(sim(X)[:,true_label])
        text_chain, text_indices_chain, perturbations_indices_chain, perturbations_chain = [], [], [], []
        for p,i in zip(perturbation_indices, range(len(perturbation_indices))):
            text_chain.append(text[p]); text_indices_chain.append(p)
            perturbations_indices_chain.append(word2index[neighbors[text[p]][permutations[ii][i]]])
            perturbations_chain.append(neighbors[text[p]][permutations[ii][i]])

        effective = True
        num_perturbations = len(perturbation_indices)
        l2norm_best_attack = np.linalg.norm(index2embedding[word2index[text_chain[0]]]-index2embedding[word2index[perturbations_chain[0]]])
        l2norm_best_attack /= normalization

        print("[logger]: Attack found")
        print("\t  True class {} confidence drops to {}".format(true_label, np.min(sim(X)[:,true_label])))
        print("\t  Indices chain {} substituted with {}".format(text_indices_chain, perturbations_indices_chain))
        print("\t  Words chain {} substituted with {}".format(text_chain, perturbations_chain))
        #if v.depth == 1:
            #print("\t  List of all single perturbations {}".format(single_perturbations))
    Q = np.max(y-sim(X)[:,true_label])  # can use 'mean', 'max'
    v.Q_v_prime = Q/self.n_sims

    return effective, num_perturbations, l2norm_best_attack


def __override_simulate_single_node_torch(self, v, sim, args):
    """
    Torch version of the __override_simulate_single_node function, specific to LSTMs.
    """
    #print("[logger]: Processing node {} at depth {}".format(v.index, v.depth))
    neighbors, x, y, true_label, text, n_size, word2index, index2word, index2embedding, normalization = args[0], cp.copy(args[1]), args[2], args[3], cp.copy(args[4]), args[5], args[6], args[7], args[8], args[9]
    if v.children != None:
        raise Exception("Expanding Exception: node at depth {} with index {} has children hence it can't be expanded".format(v.depth, v.index))
    # extract the series of perturbations to sample
    perturbation_indices = []
    tmp = v
    while tmp.parent != None:
        perturbation_indices.append(tmp.index)
        # uncomment this if you want a fast proxy to to try multiple perturbations without exploring the tree
        #p_1 = np.random.randint(0, len(text))
        #perturbation_indices.append(p_1)
        tmp = tmp.parent
    # compute simulation indices
    permutations = np.random.randint([0 for _ in range(len(perturbation_indices))],
                                     [len(neighbors[text[v]]) for v in perturbation_indices], 
                                     [self.n_sims, len(perturbation_indices)])

    # create a single vector that exploit SIMD forward of the neural network
    X = np.tile(x, (self.n_sims, 1, 1)).reshape(self.n_sims, x.shape[1], x.shape[2])
    for i in range(self.n_sims):
        for (j,n) in zip(perturbation_indices, range(len(perturbation_indices))):
            #print("Sim {} , mutating word {} at index {} with {}".format(i, text[j], j ,permutations[i,n]))
            X[i,j] = index2embedding[word2index[neighbors[text[j]][permutations[i,n]]]]/normalization  # normalize to be consistent
    #print("[logger] At index {} max perturbation leaded to {} to class 0".format(v.index, np.min(sim(X)[:,true_label])))
    X = X.reshape(self.n_sims, x.shape[1], x.shape[2])
    X = [torch.Tensor(x).unsqueeze(0) for x in X]
    effective, num_perturbations = False, 0
    worst_accuracy_drop = np.min(sim(X)[:,true_label,np.newaxis] - sim(X))
    l2norm_best_attack = 0.
    if worst_accuracy_drop < 0:
        # report all label-changing words of length 1
        single_perturbations = []
        if v.depth == 1:
            for i in range(len(permutations)):
                a = sim(X[i].reshape(1, x.shape[1], x.shape[2]))
                if a[0,true_label] - np.max(a) < 0.:
                    single_perturbations.append(neighbors[text[v.index]][permutations[i][0]])        
        ii = np.argmin(sim(X)[:,true_label])
        text_chain, text_indices_chain, perturbations_indices_chain, perturbations_chain = [], [], [], []
        for p,i in zip(perturbation_indices, range(len(perturbation_indices))):
            text_chain.append(text[p]); text_indices_chain.append(p)
            perturbations_indices_chain.append(word2index[neighbors[text[p]][permutations[ii][i]]])
            perturbations_chain.append(neighbors[text[p]][permutations[ii][i]])

        effective = True
        num_perturbations = len(perturbation_indices)
        l2norm_best_attack = np.linalg.norm(index2embedding[word2index[text_chain[0]]]-index2embedding[word2index[perturbations_chain[0]]])
        l2norm_best_attack /= normalization

        print("[logger]: Attack found")
        print("\t  True class {} confidence drops to {}".format(true_label, np.min(sim(X)[:,true_label])))
        print("\t  Indices chain {} substituted with {}".format(text_indices_chain, perturbations_indices_chain))
        print("\t  Words chain {} substituted with {}".format(text_chain, perturbations_chain))
        #if v.depth == 1:
            #print("\t  List of all single perturbations {}".format(single_perturbations))
    Q = np.mean(y-sim(X)[:,true_label])
    v.Q_v_prime = Q/self.n_sims

    return effective, num_perturbations, l2norm_best_attack
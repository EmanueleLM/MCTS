from math import log, sqrt, inf
import copy as cp
import numpy as np  # --> exploit SIMD when it is possible


class Vertex(object):
    def __init__(self, depth, parent, index, is_root=False):
        self.depth = depth
        self.is_root = is_root
        self.parent = parent
        self.children = None
        self.index = index  # i-th child
        # UTC parameters (Q(v), Q(v'), N(v), N(v'))
        self.Q_v_prime = 0.
        self.N_v = (self.parent.N_v_prime if self.parent!=None else 1)  # number of times the parent has been visited so far
        self.N_v_prime = 1  # number of times a node is visited (1 at the beginning)
    
    def heuristic(self, discount=1.):
        return self.Q_v_prime/self.N_v_prime + discount*sqrt(2*log(self.N_v)/self.N_v_prime)

class MCTree(object):
    def __init__(self, branch_factor, max_depth, n_sims=1000, l_rate=1e-3, discount=0.5):
        self.branch_factor = branch_factor
        self.max_depth = max_depth
        self.actual_depth = 0
        self.root = Vertex(depth=0, parent=None, index=0, is_root=True)
        self.V = {0: {0: [self.root]}}  # nested dictionary, one for every layer
        self.T = [self.root]  # list of terminal nodes (i.e. they can be expanded)
        self.best_node = self.root  # this is updated at every backprop iteration
        self.n_sims = n_sims
        self.l_rate = l_rate  # rate for the deltaQ update
        self.discount = discount  # UCT discount

    def select(self, discount=1.):
        # descend the tree by selecting the path whose node maximizes some heuristic
        # check at the end that the node that you expand is terminal
        if len(self.T) == 0:
            raise Exception("No Node To Expand Exception: list of terminal nodes is empty.")
        v_star, max_h = None, -inf
        for t in self.T:
            if t.heuristic(discount) > max_h:
                v_star = t
                max_h = t.heuristic(discount)
        print("T UCT")
        print([(t.depth, t.heuristic(discount)) for t in self.T])                
        print(v_star.depth, v_star.heuristic(discount))
        upd = v_star
        upd.N_v_prime +=1
        while upd.parent != None:
            upd.N_v += 1
            upd = upd.parent
        return v_star
        
    def expand(self, v):
        if v.children != None:
            raise Exception("Already Expanded Exception: vertex {} at depth {} has already been expanded.".format(v.index, v.depth))
        children = []
        children_depth = v.depth+1
        # set the new tree's depth, if changed
        if self.actual_depth < children_depth: 
            print("[logger]: Expanding the tree with a new layer, depth {}",format(children_depth))
            self.actual_depth = children_depth
        for i in range(self.branch_factor):
            v_child = Vertex(depth=children_depth, parent=v, index=i, is_root=False)
            children.append(v_child)
        # check if this is the first time the branch is visited
        if children_depth not in self.V:
            self.V[children_depth] = {}
        self.V[children_depth][v.index] = children    
        self.actual_depth = max(self.actual_depth, children_depth)
        v.children = children
        # remove the expanded node from the list of terminals and add the new children
        if v in self.T:
            self.T.remove(v)
        for c in children:
            self.T.append(c)

    def __simulate_single_node(self, v, sim, args):
        """
        Execute n_sims of simulation from node v: sim function is executed
         taking any argument (in args) and returning a floating point which is the update
         dQ to backpropagate.
        Ideally args will have some reference to a variable that is unique of the vertex,
         hence you can put into args references to the vertex index, for example. 
        """
        if v.children != None:
            raise Exception("Expanding Exception: node at depth {} with index {} has children hence it can't be expanded".format(v.depth, v.index))
        Q = .0
        for _ in range(self.n_sims):
            Q += sim(args)  
        v.Q_v_prime = Q/self.n_sims

    def simulate(self, v, sim, args, simd=False):
        """
        Execute n_sims simulations wiht function sim of every child of node v
        TODO: implement SIMD support for simulation of batches.
        """
        for c in v.children:
            if simd is False:
                self.__simulate_single_node(c, sim, args)  
            else:
                raise Exception("Not Implemented Exception.")

    def backpropagate(self, v):
        """
        Backrpop values of v's children, where v is the node that has been expanded.
        """
        ptr_v = v
        ptr_child_v = v.children
        while ptr_child_v!=None and ptr_v!=None:  # terminate when root is reached (and evaluated)
            #print("ptr_v: ", ptr_v, " ptr_child_v: ", ptr_child_v)
            dQ = 0.
            for c in ptr_child_v:
                dQ += c.Q_v_prime
            ptr_v.Q_v_prime = dQ  # update
            # move up in the tree
            ptr_child_v = (None if ptr_v.parent==None else v.parent.children)
            ptr_v = ptr_v.parent

    def adjacency(self):
        V = []
        for key1 in self.V.keys():
            for key2 in self.V[key1]:
                for v in self.V[key1][key2]:
                    V.append(v)
        adj = np.eye(len(V), len(V))
        for v in V:
            if v.children != None:
                pass
                # stuff here...
            adj[v.depth, v.index] = 1
        del V  # this matrix can be big
        return adj


def MCTS(branch_factor, max_depth, sim, n_sims=1000, l_rate=1e-3, discount=0.5):
    """
    This function implements all the routines of MCTS.
    """
    raise NotImplementedError("Function has not been implemented (yet?)")

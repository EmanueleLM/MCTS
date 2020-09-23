"""
Example on how to use create an MCTS tree
"""
from numpy.random import rand

from MCTS import MCTree

branch_factor = 2
max_depth = 10
n_sims=1000
l_rate=1e-3
discount=0.5

# 0.create the tree object
tree = MCTree(branch_factor, max_depth, n_sims, l_rate, discount) 

# 1.select (the node to expand)
v = tree.select()
assert v == tree.root, "Error: node selected must be the root."
# 2.expand
tree.expand(v)
assert len(v.children)== branch_factor, "Error: children of node must be {}.".format(branch_factor)
# 3.simulate
tree.simulate(v, rand, 1, False)
# 4.backpropagate
tree.backpropagate(v)

# execute another selection and expansion to check values of N(v) and N(v')
v_star = tree.select()
assert v_star in tree.T, "Error: node selected is not terminal."
tree.expand(v_star)
tree.simulate(v_star, rand, 1, False)
tree.backpropagate(v_star)

# execute a simulation until max_depth is reached
tree = MCTree(branch_factor, max_depth, n_sims, l_rate, discount) 
v_star = tree.root
while v_star.depth!=max_depth:
    v_star = tree.select()
    print("Node selected {} (depth={}), whose parent is {}".format(v_star, v_star.depth, v_star.parent))
    assert v_star in tree.T, "Error: node selected is not terminal."
    tree.expand(v_star)
    tree.simulate(v_star, rand, 1, False)
    tree.backpropagate(v_star)
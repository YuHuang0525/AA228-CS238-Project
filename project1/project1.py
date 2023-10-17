import sys

import networkx as nx

import numpy as np
import csv
from scipy.special import gammaln

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))



def prior(vars, G):
    n = len(vars)
    r = [vars[i]['r'] for i in range(n)]
    q = [np.prod([r[j] for j in G.predecessors(i)]) for i in range(n)]
    
    prior_matrix = [np.ones((int(q[i]), r[i])) for i in range(n)]
    return prior_matrix


def sub2ind(siz, x):
    k = np.concatenate([np.array([1]), np.cumprod(siz[:-1])])
    return int(np.dot(k, x - 1) + 1)

def statistics(vars, G, D):
    n = D.shape[0]
    r = [vars[i]['r'] for i in range(n)]
    q = [np.prod([r[j] for j in list(G.predecessors(i))]) for i in range(n)]
    M = [np.zeros((int(q[i]), r[i])) for i in range(n)]
    for o in range(D.shape[1]):
        for i in range(n):
            k = D[i, o]
            parents = list(G.predecessors(i))
            j = 1
            if parents:
                j = sub2ind([r[p] for p in parents], np.array([D[parent, o] for parent in parents]))

            M[i][j - 1, k - 1] += 1.0
    
    return M



def bayesian_score_component(M, alpha):
    p = np.sum(gammaln(alpha + M))
    p -= np.sum(gammaln(alpha))
    p += np.sum(gammaln(np.sum(alpha, axis=1)))
    p -= np.sum(gammaln(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    return p

def bayesian_score(vars, G, D):
    n = len(vars)
    M = statistics(vars, G, D)  # Assuming you have a statistics function
    alpha = prior(vars, G)     # Assuming you have a prior function
    return np.sum([bayesian_score_component(M[i], alpha[i]) for i in range(n)])

def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
            
    
    # read the variables and data from csv file
    variables = None
    data = []
    names2idx = {}

    # Open the CSV file
    with open(infile, "r") as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)

        # get the variables and construct names2idx
        variables = next(csv_reader)
        names2idx = {v: idx for idx, v in enumerate(variables)}

        # Read the remaining rows (data) and store them in the 2D matrix
        for row in csv_reader:
            data.append(row)
        
        # now we want to convert [[3,3,2,3,1,3], [1,3,2,3,2,3]] to [[3,1], [3,3], [2,2], [3,3], [1,2], [3,3]]
        D = [[] for _ in range(len(variables))]
        for row in data:
            for col in range(len(row)):
                D[col].append(int(row[col]))

    
        # build graph from the gph file 

        # create empty graph
        G = nx.DiGraph()
        # read nodes from .gph file and add to graph
        with open(outfile, "r") as file:
            for line in file:
                parent, child = line.strip().split(",")
                G.add_edge(names2idx[parent], names2idx[child])
        
        # adjust variable list
        vars = [{"symbol": key, "r":3} for key in variables]        

        
    # print(G.edges)
    # print(G.nodes)
    # print(vars)
    return bayesian_score(vars, G, np.array(D))


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    print(compute(inputfilename, outputfilename))


if __name__ == '__main__':
    main()

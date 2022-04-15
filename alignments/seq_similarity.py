#/usr/bin/python
import os
import sys
from leven import levenshtein       
import numpy as np
from sklearn.cluster import dbscan
import argparse
from itertools import groupby 

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-seq", dest="seq", required=True,
                        help="The .seq file from silentsequence")
    parser.add_argument("-dist_cutoff", dest="dist", type=int, default=5,
                        help="Levenshtein string distance cutoff for clustering")
    parser.add_argument("-debug", dest="debug", default=False,
                        help="Whether dump output")
    args = parser.parse_args()
    return args

args = get_args()


def load_seq(filepath):
    seqs = set()
    seqname_pair = {}
    with open(filepath,'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            seq = line.split(" ")[0]
            if seq in seqs:
                continue
            seqs.add(seq)
            name = line.split(" ")[-1]
            seqname_pair[seq] = name
    f.close()
    return seqname_pair

def lev_metric(x, y):
    i, j = int(x[0]), int(y[0])     # extract indices
    return levenshtein(data[i], data[j])

def find_cluster_center(labels):
    ## Nodes: [0,1,2,...n]
    ## labels: [-1,-1,0,1,2,...]
    noise_nodes = np.where(labels == -1)[0]
    tmp_nodes = np.unique(labels)
    unique_nodes = np.delete(tmp_nodes, np.where(tmp_nodes == -1))
    clusters = []
    for i in range(len(unique_nodes)):
        ## Get the first occurence as cluster center
        clusters.append(np.where(labels == i)[0][0])
    return clusters, list(noise_nodes) 
 


######### Main ###########

seqname_pair = load_seq(args.seq)
seqs = list(seqname_pair.keys())
print("############### Begin sequence clustering ###############")
print("Total unique sequences:", len(seqs))
seqs.sort(key=lambda s: len(s))
grouped_seqs = [list(g) for k, g in groupby(seqs, key=len)]
print("Total length groups:", len(grouped_seqs))
#print(grouped_seqs) 

clustered_seqs = []
for data in grouped_seqs:  
    
    X = np.arange(len(data)).reshape(-1, 1) 
    nodes, labels = dbscan(X, metric=lev_metric, eps=args.dist, min_samples=2)
    ## Unsure clusters will all be regarded as -1, so you need treat them separately
    clusters, noise_nodes = find_cluster_center(labels)

    if args.debug:
        print("---------- ")
        #print(data)
        print(len(data)) 
        #print(len(data[0])) 
        print(np.unique(labels)) 
        print("Cluster: ", clusters) 
        print("noise_nodes", noise_nodes)
        print("Combined clusters:", len(clusters)+len(noise_nodes)) 
     
    all_clusters = clusters + noise_nodes

    for i in all_clusters:
        clustered_seqs.append(data[i])  
 
print("\n############### Complete sequence clustering ###############")
print("Total output clusters:", len(clustered_seqs)) 
with open("clustered.seq",'w') as f:
    for i in clustered_seqs:
        f.write("%s %s\n"%(i, seqname_pair[i]))
f.close()
    


    

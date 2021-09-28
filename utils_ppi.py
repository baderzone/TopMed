#!/usr/bin/env python
# coding: utf-8


import numpy as np
import collections
import copy
import pdb

from collections import Counter


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def count_edges(network):
    # counts directed edges
    n = 0
    for a in network.keys():
        for b in network[a]:
            n += 1
    return(n)


def make_undirected(network):
    ret = copy.deepcopy(network)
    for a in network.keys():
        for b in network[a]:
            if a not in ret[b]:
                ret[b].update([a])
    n1 = count_edges(network)
    n2 = count_edges(ret)
    logger.info(
        'directed network with %d edges -> undirected network with %d edges', n1, n2)
    return(ret)


def read_ppi(filename):
    fp = open(filename, 'r')
    ret = collections.defaultdict(set)
    next(fp)  # skip header line
    for line in fp:
        toks = line.strip().split()
        assert(len(toks) == 3), 'bad line: ' + line
        (a, b) = (toks[1], toks[2])
        ret[a].update([b])

    fp.close()
    n = count_edges(ret)
    logger.info('%s -> %d undirected interactions', filename, n)
    return(ret)

# network is a dict of sets, vertex to its neighbors
# u_to_h is a dict of sets, uniprot to hgncs


def convert_uniprot_to_hgnc(network, u_to_h):
    ret = collections.defaultdict(set)
    for u1 in network:
        if u1 not in u_to_h:
            # logger.info('no uniprot for %s', u1)
            continue
        set1 = u_to_h[u1]
        for u2 in network[u1]:
            if u2 not in u_to_h:
                # logger.info('no uniprot for %s', u2)
                continue
            set2 = u_to_h[u2]
            for a in set1:
                if a not in ret:
                    ret[a] = set()
                for b in set2:
                    ret[a].add(b)
    return(ret)


def gene_protein(filename):
    # Find mapping of protein name to Gene names
    fp = open(filename, 'r')
    ret = collections.defaultdict(list)
    next(fp)  # skip header line
    for line in fp:
        toks = line.strip().split()

        (a, b) = (toks[0], toks[2:])
        for gene in b:
            ret[gene].append(a)
    return(ret)


def protein_gene(filename):
    fp = open(filename, 'r')
    ret = collections.defaultdict(list)
    next(fp)
    for line in fp:
        toks = line.strip().split()
        (a, b) = (toks[0], toks[2:])
        for gene in b:
            ret[a].append(gene)
    return (ret)


def gene_gene(gene_list, protein_gene_dict, gene_protein_dict, protein_protein_dict):
    gene_gene_dict = collections.defaultdict(list)
    for gene in gene_list:
        protein1_list = []
        protein1_list.extend(gene_protein_dict[gene])
        protein2_list = []
        for x in protein1_list:
            protein2_list.extend(protein_protein_dict[x])
        gene2_set = set()
        for y in protein2_list:
            gene2_set = gene2_set.union(set(protein_gene_dict[y]))
        gene2_nw = set(gene2_set).intersection(set(gene_list))
        gene2_nw.discard(gene)
        if len(gene2_nw) > 0:
            gene_gene_dict[gene].extend(list(gene2_nw))

    return (gene_gene_dict)


# how many undericted edges within a set of vertices?
def count_undirected_within_set(myset, network_interactions):
    n = 0
    for a in myset:

        if a not in network_interactions:
            continue
        for b in network_interactions[a]:
            if (b < a):
                continue
            if b not in myset:
                continue
            n += 1
    return(n)

# how many dericted edges within a set of vertices?


def count_directed_within_set(myset, network_interactions):
    n = 0
    for a in myset:

        if a not in network_interactions:
            continue
        for b in network_interactions[a]:
            if b not in myset:
                continue
            n += 1
    return(n)

# how many edges between one vertex and a set of vertices?


def count_vertex_to_set(v, myset, network_interactions):
    n = 0
    if v in network_interactions:
        for a in network_interactions[v]:
            if a in myset:
                n += 1
    return(n)


def vertex_to_set(v, myset, network_interactions):
    connected = set()
    if v in network_interactions:
        for a in network_interactions[v]:
            if a in myset:
                connected.add(a)
    return(connected)


# calculate fraction of occupied edges between genes for an undirected network
# don't include self-edges
def get_fnull(genes, network_interactions):
    logger.info('calculating fnull')
    nedge = 0
    mylist = sorted(genes)
    for v in mylist:
        if v not in network_interactions:
            continue
        for w in mylist:
            # skip self-edges
            if (w >= v):
                continue
            if w in network_interactions[v]:
                nedge += 1

    nvert = len(mylist)
    npair = float(0.5 * nvert * (nvert - 1.0))
    fnull = float(nedge) / float(npair)
    # the average degree should be (fnull)(nvert - 1)
    avgdegree = fnull * (nvert - 1.0)
    logger.info('ngenes %d nedge %d npair %d fnull %f avgdeg %f',
                nvert, nedge, npair, fnull, avgdegree)
    return(fnull)


def get_degree(geneset, network_interactions):
    logger.info('computing degree of genes')
    gene_degree = dict()
    mylist = sorted(geneset)
    for v in mylist:
        ndegree = 0
        if v not in network_interactions:
            gene_degree[v] = 0
            continue
        for w in network_interactions[v]:
            if (w not in geneset):
                continue
            ndegree += 1
        gene_degree[v] = ndegree

    return gene_degree


def get_D_other(otherset, gene_degree):
    D1 = 0
    D2 = 0
    for g in otherset:
        D1 += gene_degree[g]
        D2 += gene_degree[g]**2
    return D1, D2


def get_lambda_gene(gene, gene_degree, D1, D2, c):
    f_g = gene_degree[gene]
    lambda_gene = (1 / (2 * c)) * ((D1 + f_g)**2 - (D2 + f_g**2))
    return lambda_gene


def Trrust_dict(filename):
    fp = open(filename, 'r')
    TFsource_target = dict()
    TFsource_target_sign = dict()
    next(fp)  # skip header line
    for line in fp:
        toks = line.strip().split()
        assert(len(toks) == 4), 'bad line: ' + line
        (a, b, c) = (toks[0], toks[1], toks[2])
        if not a in TFsource_target:
            TFsource_target[a] = set()
            TFsource_target_sign[a] = dict()
        TFsource_target[a].update([b])
        TFsource_target_sign[a][b] = c
    return (TFsource_target, TFsource_target_sign)


def reverse_network(network_interactions):
    logger.info('reversing network direction')
    reverse_interactions = collections.defaultdict(set)
    for v in network_interactions:
        for w in network_interactions[v]:
            reverse_interactions[w].update([v])
    return reverse_interactions


def get_directedD_other(otherset, gene_outdegree, gene_indegree):
    Din = 0
    Dout = 0
    DinDout = 0

    for g in otherset:
        Din += gene_indegree[g]
        Dout += gene_outdegree[g]
        DinDout += gene_indegree[g] * gene_outdegree[g]
    return Din, Dout, DinDout


def get_directedlambda_gene(gene, gene_outdegree, gene_indegree, Din, Dout, DinDout, c):
    gene_din = gene_indegree[gene]
    gene_dout = gene_outdegree[gene]
    lambda_gene = (1 / c) * (gene_din * Dout +
                             gene_dout * Din + Din * Dout - DinDout)
    return lambda_gene


# def get_TFlambda_gene(otherset, network_interactions, reverse_interactions, c, gene=set()):
#     myset = otherset.union([gene])
#     gene_outdegree = get_outdegree(myset, network_interactions)
#     gene_indegree = get_indegree(myset, reverse_interactions)
#     lambda_gene = 0
#     Dout = 0
#     for v in gene_outdegree:
#         Dout = gene_outdegree[v]
#         if Dout == 0:
#             continue
#         Din = 0
#         for w in gene_indegree:
#             Din += gene_indegree[w]

#         lambda_gene += Dout * Din / c

#     return lambda_gene

# def get_degcor_lambdanull(gene_nulldegree, c):
#     D1 = 0
#     D2 = 0
#     for gene in gene_nulldegree:
#         D1 += gene_nulldegree[gene]
#         D2 += gene_nulldegree[gene]**2

#     pdb.set_trace()
#     lambda_null = (1 / (2 * c)) * (D1**2 - D2)

#     return lambda_null

# def get_gene_lambda(gene, otherset, network_interactions, c):
#     othergene_degree = get_genedegree(otherset, network_interactions)
#     D1 = 0
#     D2 = 0
#     for g in othergene_degree:
#         D1 += othergene_degree[g]
#         D2 += othergene_degree[g]**2

#     # pdb.set_trace()
#     gene_degree = count_vertex_to_set(gene, otherset, network_interactions)
#     pdb.set_trace()
#     gene_lambda = (1 / (2 * c)) * ((D1 + 2 * gene_degree)
#                                    ** 2 - (D2 + (2 * gene_degree)**2))

#     return gene_lambda


# def gene_ppiscore(gene, other_genes, network_interactions,  xnull):
#     # get score of candidate gene
#     # keep union so when add >1 gene it works
#     new_set = other_genes.union({gene})
#     x = count_undirected_within_set(new_set, network_interactions)
#     if x > 0:
#         score = x * np.log(x / xnull) - (x - xnull)
#     if x == 0:
#         score = 0  # this could happen in some random initializations
#     return (score)


# def get_indegree(geneset, reverse_interactions):
#     gene_degree = dict()
#     mylist = sorted(geneset)
#     for v in mylist:
#         ndegree = 0
#         if v not in reverse_interactions:
#             gene_degree[v] = 0
#             continue
#         for w in reverse_interactions[v]:
#             if (w not in geneset):
#                 continue
#             ndegree += 1
#         gene_degree[v] = ndegree
#     return gene_degree

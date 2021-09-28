#!/usr/bin/env python
# coding: utf-8


import numpy as np
from scipy.stats import chi2_contingency


def get_pathway_genes(filename):
    pathway_dict = {}
    fp = open(filename, 'r')
    for line in fp:
        toks = line.strip().split('\t')
        assert toks[0] not in pathway_dict, 'repeated pathway: ' + toks[0]
        pathway_dict[toks[0]] = set(toks[2:])
    fp.close()
    return (pathway_dict)


def pathway_ranks(genes, pathway_dict, pathway_gene_universe, alpha):
    ret = {}
    for pathway in pathway_dict.keys():
        gene_in_both = pathway_dict[pathway].intersection(set(genes))
        count_11 = len(gene_in_both)
        if count_11 >= 5:
            count_11 = len(gene_in_both)
            count_12 = len(genes) - count_11
            count_21 = len(pathway_dict[pathway]) - count_11
            count_22 = len(pathway_gene_universe) - (count_12 + count_21)
            table = np.array([[count_11, count_12], [count_21, count_22]])
            g, p, dof, expctd = chi2_contingency(table)
            if p < alpha:
                ret[pathway] = p

    sorted_ret = {kk: v for kk, v in sorted(ret.items(), key=lambda v: v[1])}
    return sorted_ret


def gene_pathway_score(current_pathway_ranks, candidate_gene, pathway_genes, Lk_network_genes, other_genes, k):
    ret = 0
    pathway_genes_exhausted = set()
    for pathway in current_pathway_ranks:
        genes_in_pathway = set(pathway_genes[pathway])
        genes_in_pathway.difference_update(pathway_genes_exhausted)

        candidate_network_genes = set(other_genes).union({candidate_gene})
        common_genes = candidate_network_genes.intersection(genes_in_pathway)
        pathway_genes_exhausted.update(genes_in_pathway)

        Lk_pathway_genes = set(Lk_network_genes).intersection(genes_in_pathway)
        lambda_null = len(Lk_pathway_genes) / k
        n_11 = len(common_genes)
        if n_11 == 0:
            current_score = lambda_null
        else:
            current_score = (n_11 * np.log(n_11 / lambda_null)
                             ) - (n_11 - lambda_null)
        ret += (current_score)

    return ret


def network_pathwayscore(current_pathway_ranks, pathway_genes, Lk_network_genes, network_genes, k):
    ret = 0
    pathway_genes_exhausted = set()
    for pathway in current_pathway_ranks:
        genes_in_pathway = set(pathway_genes[pathway])
        genes_in_pathway.difference_update(pathway_genes_exhausted)

        candidate_network_genes = set(network_genes)
        common_genes = candidate_network_genes.intersection(genes_in_pathway)
        pathway_genes_exhausted.update(genes_in_pathway)

        Lk_pathway_genes = set(Lk_network_genes).intersection(genes_in_pathway)
        lambda_null = len(Lk_pathway_genes) / k
        n_11 = len(common_genes)
        if n_11 == 0:
            current_score = lambda_null
        else:
            current_score = (n_11 * np.log(n_11 / lambda_null)
                             ) - (n_11 - lambda_null)
        ret += (current_score)

    return ret


def locus_pathway_scores(geneset, current_pathway_ranks, pathway_genes, Lk_network_genes, other_genes, k):
    ret = []
    for candidate_gene in geneset:
        # get score of candidate gene
        score = gene_pathway_score(
            current_pathway_ranks, candidate_gene, pathway_genes, Lk_network_genes, other_genes, k)
        ret.append(score)  # (g, score))
    return (ret)

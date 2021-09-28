import argparse
import yaml
import logging
import sys
import numpy as np
import random
import utils_data
import utils_ppi
import utils_pathway
import pandas as pd
import os as os
import copy
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import graphviz as gv
from scipy.stats import chi2_contingency
from scipy.special import gammaln
from scipy import optimize
from scipy.stats import cauchy
from scipy.stats import iqr

import pdb

logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('gene_selection_simple')
logger.setLevel(logging.INFO)



def get_args():
    # ,epilog='small call: see run.sh', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description='Select genes')
    parser.add_argument('-c', '--config', help='config file',
                        required=False, type=str, default='./run_geneselect.yaml')
    parser.add_argument('-p', '--phenotype_list', help='phenotype list',
                        required=False, type=list, default=['QT', 'HR', 'QRS', 'PR', 'JT'])
    parser.add_argument('-k', '--k_closestgenes', help='number of closest genes',
                        required=False, type=int, default=1)
    parser.add_argument('-d', '--flankdistance', help='distance range from which to pick genes',
                        required=False, type=int, default=250000)
    parser.add_argument('-a', '--alpha', help='pathway score significance',
                        required=False, type=float, default=1e-10)
    parser.add_argument('-s0', '--seed_start', help='seed0',required=False, type=int, default=0)
    parser.add_argument('-s1', '--seed_end', help='seed1',required=False, type=int, default=1)
    parser.add_argument(
        '--nodegcor', help='degree corrected null', action='store_true')  # default is False
    parser.add_argument(
        '--plot', help='plot network', action='store_true')  # default is False
    parser.add_argument(
        '--nooptim', help='degree corrected null', action='store_true')  # default is False

    args = parser.parse_args()
    return(args)


class Config():
    # generate run configuration object from user-provided config file in yaml format
    def __init__(self, path):
        f_h = open(path, 'r')
        config = yaml.safe_load(f_h)
        f_h.close()

        self.__dict__ = config
        logger.info('config %s', str(self.__dict__))
        return


def write_pheno_locus_genes(filename, phenotype, pheno_locus_to_genes):
    logger.info('writing %s', filename)
    with open(filename, 'w') as f:
        f.write('\t'.join(['phenotype', 'locus', 'gene', 'distance(kbp)']) + '\n')
        for locus, gene_distance in pheno_locus_to_genes.items():
            for (g, d) in gene_distance:
                f.write('\t'.join([phenotype, locus, g, str(d)]) + '\n')
    return None


def write_locus_genes(filename, locus_gene_dist, gene_phenoset):
    logger.info('writing %s', filename)
    with open(filename, 'w') as f:
        f.write('\t'.join(['locus', 'gene', 'pheno', 'distance(kbp)', 'ngenes']) + '\n')
        # sort genes by distance and then by name
        loci = sorted(locus_gene_dist.keys())
        for locus in loci:
            tuples = locus_gene_dist[locus].items()
            mysort = sorted(tuples, key=lambda x: (x[1], x[0]))
            ngenes = len(mysort)
            for (g, d) in mysort:
                phenostr = '+'.join(sorted(gene_phenoset[g]))
                f.write('\t'.join([locus, g, phenostr, str(d), str(ngenes)]) + '\n')
    return None


def write_snp_genes(filename, snp_gene_distance, gene_phenoset):
    logger.info('writing %s', filename)

    with open(filename, 'w') as f:
        f.write('\t'.join(['snp', 'gene', 'phenotype', 'distance(kbp)', 'ngenes']) + '\n')
        # sort genes by distance and then by name
        snp_list = sorted(snp_gene_distance.keys())
        for snp in snp_list:
            tuples = snp_gene_distance[snp].items()
            mysort = sorted(tuples, key=lambda x: (x[1], x[0]))
            ngenes = len(mysort)
            for (g, d) in mysort:
                phenostr = '+'.join(sorted(gene_phenoset[g]))
                f.write('\t'.join([snp, g, phenostr, str(d), str(ngenes)]) + '\n')
    return None


def write_locus_activeset(filename, locus_activeset, gene_special):
    logger.info('writing %s', filename)
    with open(filename, 'w') as f:
        f.write('\t'.join(['locus', 'selected_gene', 'gene_special']) + '\n')
        for locus in sorted(locus_activeset.keys()):
            for gene in sorted(locus_activeset[locus]):
                specialstr = ''
                if gene_special[gene]:
                    specialstr = '+'.join(sorted(list(gene_special[gene])))
                f.write('\t'.join([locus, gene, specialstr]) + '\n')
    return None


def get_gene_distance(snp_gene_distance):
    gene_distance = dict()
    # first find the closest distance
    for snp in snp_gene_distance:
        for gene in snp_gene_distance[snp]:
            d = snp_gene_distance[snp][gene]
            if not gene in gene_distance:
                gene_distance[gene] = d
            elif (d < gene_distance[gene]):
                gene_distance[gene] = d
    return (gene_distance)


def get_gene_signeddistance(snp_gene_signeddist):
    gene_distance = dict()
    # first find the closest distance
    for snp in snp_gene_signeddist:
        for gene in snp_gene_signeddist[snp]:
            d = snp_gene_signeddist[snp][gene]
            if not gene in gene_distance:
                gene_distance[gene] = d
            elif (abs(d) < abs(gene_distance[gene])):
                gene_distance[gene] = d
    return (gene_distance)


def get_gene_closestsnps(snp_gene_distance, gene_distance):
    gene_closestsnps = dict()
    for gene in gene_distance:
        gene_closestsnps[gene] = set()
        d0 = gene_distance[gene]
        for snp in snp_gene_distance:
            if gene in snp_gene_distance[snp]:
                if snp_gene_distance[snp][gene] == d0:
                    gene_closestsnps[gene].add(snp)

        assert(len(gene_closestsnps[gene]) >
               0), 'error, did not find a closest snp for gene %s' % gene
    return(gene_closestsnps)


def write_gene_closest(filename, gene_distance, gene_closestsnps):
    logger.info('writing gene_closestto %s', filename)
    fp = open(filename, 'w')
    fp.write('\t'.join(['gene', 'distance(kbp)', 'closest_snps']) + '\n')
    for gene in gene_distance:
        d = gene_distance[gene]
        snpstr = ';'.join(sorted(gene_closestsnps[gene]))
        fp.write('\t'.join([gene, str(d), snpstr]) + '\n')
    fp.close()
    return None


def plot_maxdistances(snp_gene_distance):
    max_distances = []
    for snp in snp_gene_distance:
        # k_genes = list(snp_gene_distance[snp].keys())[0:k]
        dist = []
        for gene in snp_gene_distance[snp]:
            dist.append(snp_gene_distance[snp][gene])
        max_dist = np.max(dist)
        max_distances.append(max_dist)

    plt.hist(max_distances)
    plt.savefig('./results/max_distanes.png')
    plt.close()
    return None


def activeset_mindistance(locus_geneset, gene_distance):
    print('activeset based on distance')
    locus_activeset = defaultdict()
    for locus in locus_geneset:
        recs = [(g, gene_distance[g]) for g in locus_geneset[locus]]
        genes, distances = zip(*((recs)))
        min_distance = np.min(distances)
        min_dist_genes = {x[0] for x in recs if x[1]
                          == min_distance}  # could be more than one
        for gene in sorted(min_dist_genes): #select only one gene at random
            locus_activeset[locus] = set([gene])
            break
    return(locus_activeset)


def initialize_activeset_random(locus_geneset, gene_distance):
    print('initialize activeset randomly')
    locus_activeset = defaultdict()
    # otherwise there is randomness in set, no matter the seed!
    locuslist_sorted = sorted(locus_geneset.keys())
    for locus in locuslist_sorted:
        locus_genelist = sorted(locus_geneset[locus])
        recs = [(g, gene_distance[g]) for g in locus_genelist]
        locus_activeset[locus] = {random.choice(recs)[0]}
    return(locus_activeset)


def get_distancescore_cauchy_old(distance, params):
    a = params['distfac']
    ret = (a * a) / ((distance * distance) + (a * a))
    ret = np.log(ret)
    return(ret)

def get_distancescore_cauchy(distance, params):
    a = params['distfac_active']
    b = params['distfac_inactive']

    ret =  ((distance * distance) + (b * b)) / ((distance * distance) + (a * a))
    ret = np.log(ret)
    return(ret)


def get_specialscore(gene_functions, params):
    score = 0
    for function in gene_functions:
        score += params[function]
    ret = score
    return(ret)


def get_BIC_score(cnt, nullcnt, ONESIDED=True):
    x = float(cnt)
    y = float(nullcnt)
    ret = y
    if (x > 0.):
        ret = x * math.log(x / y) - (x - y)
    if ONESIDED and (x < y):
        ret = -1 * np.abs(ret)
    return ret


def bayesscore_poisson(e, total, lambdazero):
    ret = gammaln(e + 1.0) - (e * math.log(lambdazero)) + \
        lambdazero - math.log(total)
    return (ret)


def get_poisson_score(e, total, lambdazero):
    val = bayesscore_poisson(e, total, lambdazero)
    ret = val
    if (e < lambdazero):
        val0 = bayesscore_poisson(lambdazero, total, lambdazero)
        ret = val0 - abs(val - val0)
    return(ret)

def onelocus(locus, locus_activeset, locus_nchange, locus_activelist_series, locus_geneset,  locus_score, gene_distance, gene_special, params,  network_ppi,  xnull, scores_logfile, gene_PPIdegree, degree_correction, ppi_Ksquared, TFsource_target, reverse_TF, TF_Ksquared, gene_TFindegree, gene_TFoutdegree):

    # update the activeset for this locus in place
    # get the score for each gene

    fp = open(scores_logfile, 'a')

    logger.info('working on locus %s', locus)
    total_activegenes = 0
    for x in locus_activeset:
        total_activegenes += len(locus_activeset[x])
    logger.info('%d active genes in network', total_activegenes)

    # pairs of universe network
    totalpair = 0.5 * len(locus_activeset) * (len(locus_activeset) - 1)

    recs = []

    for gene in sorted(locus_geneset[locus]):
        score_distance = get_distancescore_cauchy(
            gene_distance[gene], params)
        #dist_str = '%.0fkb' % (float(gene_distance[gene]) / 1000.0)
        dist_str = str(float(gene_distance[gene]) / 1000.0)

        # the way to initialize gene_special is as an empty set
        score_special = 0
        for special in gene_special[gene]:
            score_special += params[special]

        score_ppi = 0.0
        edges_gene_to_others = 0  # for printing in output file
        other_edges = 0  # for printing in output file
        ppi_connected_genes_str = ''  # for printing in output file
        if network_ppi is not None:
            other_activegenes = set()
            for s in locus_activeset:
                if s == locus:
                    continue
                other_activegenes.update(locus_activeset[s])

            # total number of ppi between other_activegenes
            other_edges = utils_ppi.count_undirected_within_set(
                other_activegenes, network_ppi)
            # total number of ppi between all activegenes (includes current locus)
            current_activegenes = other_activegenes.union(
                locus_activeset[locus])
            current_edges = utils_ppi.count_undirected_within_set(
                current_activegenes, network_ppi)

            logger.info(
                '%d number of edges between other_activegenes', other_edges)
            logger.info('%d number of all edges with this locus included',
                        current_edges)


            ppi_connected_genes = set()
            edges_gene_to_others = 0
            if gene in network_ppi:
                ppi_connected_genes = network_ppi[gene].intersection(
                    other_activegenes)
            edges_gene_to_others = len(ppi_connected_genes)

            # no degree_correction:
            score_BIC_nodegcor = get_BIC_score(
                other_edges + edges_gene_to_others, xnull)



            score_ppi_nodegcor = get_poisson_score(
                other_edges + edges_gene_to_others, totalpair, xnull)

            # degree_correction:
            D1, D2 = utils_ppi.get_D_other(other_activegenes, gene_PPIdegree)
            x_gene = utils_ppi.get_lambda_gene(
                gene, gene_PPIdegree, D1, D2, ppi_Ksquared)

            #score_ppi_degcor = get_BIC_score(other_edges + edges_gene_to_others, x_gene)


            score_ppi_degcor = get_poisson_score(
                other_edges + edges_gene_to_others, totalpair, x_gene)



            tmp = []
            for g in sorted(ppi_connected_genes):
                specialstr = ''
                if gene_special[g]:
                    specialstr = '{' + \
                        ','.join(sorted(list(gene_special[g]))) + '}'
                tmp.append(g + specialstr)

            ppi_connected_genes_str = ','.join(tmp)

            if degree_correction:
                score_ppi = score_ppi_degcor
            else:
                score_ppi = score_ppi_nodegcor


        TF_ingenes_str = ''
        TF_outgenes_str = ''
        TFedges_gene_str = ''
        score_TF = 0
        score = score_distance + score_special + score_ppi + score_TF

        recs.append((score, gene))

        gene_special_str = ','.join([x for x in gene_special[gene]])
        mystr = '\t'.join([locus, gene, gene_special_str,
                           str(score_special), dist_str, str(score_distance), str(score_ppi), str(other_edges), str(edges_gene_to_others), ppi_connected_genes_str, str(gene_PPIdegree[gene]), str(score_TF), TF_ingenes_str, TF_outgenes_str, TFedges_gene_str, str(score)])
        logger.info('%s', mystr)
        fp.write(mystr + '\n')

    recs = sorted(recs)
    (bestscore, bestgene) = recs[-1]

    bestgenes = set()
    for score, gene in recs:
        if score == bestscore:
            bestgenes.add(gene)

    # sorted so randomness is reproducable!
    new_gene = random.choice(sorted(bestgenes))
    new_geneset = {new_gene}




    bestline = '\t'.join(
        [locus, 'initial: ' + str(locus_activelist_series[locus][0]),  'selected: ' + str(new_geneset)])
    logger.info('%s', bestline)
    fp.write(bestline + '\n')

    locus_nchange[locus]['changed'] = False
    if locus_activeset[locus] != new_geneset:
        locus_nchange[locus]['nchange'] += 1
        locus_nchange[locus]['changed'] = True

    locus_activeset[locus] = new_geneset

    locus_score[locus].append(bestscore)

    locus_activelist_series[locus].append(list(new_geneset)[0])  # ONLY 1
    besties = ', '.join(locus_activelist_series[locus])
    bestiesline = '\t'.join([locus, besties])
    fp.write(bestiesline + '\n')

    fp.close()

    return None




def get_active_distance(locus_activeset, gene_distance):
        dist = []
        for locus in locus_activeset:
            for gene in locus_activeset[locus]: #only one gene but this is a set
                break
            dist.append(gene_distance[gene])
        distances = np.array(dist)
        return distances

def get_inactive_distance(locus_geneset, locus_activeset, gene_distance):
        dist = []
        for locus in locus_geneset: #assume two final genes with same scores have same distance
            for active_gene in locus_activeset[locus]: #only one gene but this is a set
                break
            for gene in locus_geneset[locus]:
                if gene.startswith('locus'):
                    break
                if gene == active_gene:
                    continue
                dist.append(gene_distance[gene])

        distances = np.array(dist)
        return distances



def write_gene_distances(locus_geneset, locus_activeset, gene_signeddistance, results_dir, seed, degree_correction,  optim_run):


    filename = 'distances_final_seed' + str(seed) + '_ppi' + str(ppi_run) + '_degcor' + str(degree_correction) + '_TF' + str(TF_run) + '_optim' + str(optim_run) + '.txt'

    fp = open(filename, 'w')
    fp.write('\t'.join(['snp', 'gene', 'distance(kbp)', 'active']) + '\n')
    for locus in locus_geneset: #assume two final genes with same scores have same distance
        for active_gene in locus_activeset[locus]: #only one gene but this is a set

            break
        for gene in locus_geneset[locus]:
            if gene.startswith('locus'):
                break

            distance = gene_signeddistance[gene]
            if gene in locus_activeset[locus]: #only one active gene but this is a set
                fp.write('\t'.join([locus, gene, str(distance), 'True']) + '\n')
            else:
                fp.write('\t'.join([locus, gene, str(distance), 'False']) + '\n')
    fp.close()
    return None


def distance_func(x, distances):
    n = len(distances)
    return (1/(x**2)) - (2/n)*np.sum(1/(x**2 + distances**2))  # only one real root at x = 1



def estimate_loc(data, weight, reverse=True):
    data_weight = tuple(zip(data, weight))
    sorted_data_weight = sorted(data_weight, key=lambda x: x[1])
    cutoff = 0.5 * sum(weight)

    top_weight_sum = 0
    median = 0

    for d, w in sorted_data_weight:
        top_weight_sum = top_weight_sum + w
        if top_weight_sum >= cutoff:
            median = d
            break

    return median


def estimate_scale(data, weight):
    data_weight = tuple(zip(data, weight))
    sorted_data_weight = sorted(data_weight, key=lambda x: x[1], reverse=True)
    cutoff_lo = 0.25 * sum(weight)
    cutoff_hi = 0.75 * sum(weight)

    top_weight_sum = 0
    median_hi = 0
    median_lo = 0
    lo_assigned = False

    for d1, w in sorted_data_weight:

        top_weight_sum = top_weight_sum + w

        if (top_weight_sum >= cutoff_lo and not lo_assigned):
            median_lo = d1
            lo_assigned = True
        elif (top_weight_sum >= cutoff_hi):
            median_hi = d1
            break

    scale = 0.5 * abs(median_hi - median_lo)

    return (scale)

def initialize(active, inactive):

    both_colours = np.concatenate((active, inactive))

    data_std = np.std(both_colours)
    data_mean = np.mean(both_colours)

    active_loc_init = data_mean - data_std
    active_scale_init = 0.5 * iqr(both_colours)

    inactive_loc_init = data_mean + data_std
    inactive_scale_init = 0.5 * iqr(both_colours)

    return(both_colours, active_loc_init, active_scale_init, inactive_loc_init, inactive_scale_init)


def weight_of_colour(colour_likelihood, total_likelihood):
    """
    Compute the weight for each colour at each data point.
    """
    return colour_likelihood / total_likelihood

def onepass(locus_activeset, locus_activelist_series, locus_geneset, locus_score, locus_nchange, pass_locus_change, gene_distance, gene_special, params, network_ppi,  xnull, scores_logfile, gene_PPIdegree, degree_correction, ppi_Ksquared, TFsource_target, reverse_TF, TF_Ksquared, gene_TFindegree, gene_TFoutdegree):

    locuslist_sorted = sorted(locus_geneset.keys())
    locuslist_randomorder = random.sample(
        locuslist_sorted, len(locuslist_sorted))

    fp = open(scores_logfile, 'w')
    fp.write('\t'.join(['locus', 'gene', 'special',
                        'special_score',  'distance(kbp)', 'distance_score', 'ppi_score', 'edges_in_otheractive', 'edges_to_otheractive', 'connected_to_otheractive', 'potential_PPI', 'TF_score', 'TF_ingenes_str', 'TF_outgenes_str', 'TFconnected_to_otheractive', 'total_score']) + '\n')
    fp.close()
    for locus in locuslist_randomorder:

        logger.info('locus %s', locus)

        onelocus(locus, locus_activeset, locus_nchange, locus_activelist_series, locus_geneset, locus_score, gene_distance,
                 gene_special, params,  network_ppi,  xnull, scores_logfile, gene_PPIdegree, degree_correction, ppi_Ksquared, TFsource_target, reverse_TF, TF_Ksquared, gene_TFindegree, gene_TFoutdegree)

    npass = len(pass_locus_change)
    changed_loci = []
    for locus in locus_nchange:
        if locus_nchange[locus]['changed']:
            changed_loci.append((locus, locus_nchange[locus]['nchange']))

    pass_locus_change[npass] = changed_loci

    return None


def plot_network(number, results_dir, locus_activeset, network_ppi, phenotype_list, gene_phenoset, gene_special, degree_correction, TFsource_target):
    logger.info('plotting network')
    graph_name = 'pass' + str(number) + '_degcor' + str(degree_correction)
    img_file = results_dir + graph_name + '.pdf'
    G = gv.Digraph(graph_name, filename=img_file,
                   engine='fdp', strict=True)  # 'neato' format='png'

    # PPI  pairs
    selected_ppi = dict()
    selected_genes = set()
    for locus in locus_activeset:
        selected_genes.update(locus_activeset[locus])

    for locus in locus_activeset:
        for gene in locus_activeset[locus]:
            if gene in network_ppi:
                paired = network_ppi[gene].intersection(selected_genes)
                selected_ppi[gene] = paired

    selected_TF = dict()
    for gene in selected_genes:
        if gene in TFsource_target:
            targets = TFsource_target[gene].intersection(selected_genes)
            selected_TF[gene] = targets

    # phenotype
    G.attr('node', shape='circle', fixedsize='true',
           width='0.3', fontsize='6')
    for phenotype in phenotype_list:
        G.node(phenotype)

    # Genes - phenotype
    G.attr('node', fixedsize='true')
    for locus in locus_activeset:
        gene = list(locus_activeset[locus])[0]
        phenotypes = gene_phenoset[gene]
        special = gene_special[gene]
        color = 'white'
        if special:
            if 'omim' in special:
                color = 'red'
            elif 'lof' in special:
                color = 'orange'
            elif 'coloc' in special:
                color = 'yellow'
        G.node(gene, style='filled', fillcolor=color,
               shape='rectangle', width='0.5',  height='0.2', fontsize='6')

        # gene-phenotype edges
        if phenotypes:
            pheno = list(phenotypes)[0].split('+')
            for ph in pheno:
                G.edge(gene, ph, color='gray', arrowhead='none')

        # PPI edges
        if gene in selected_ppi:
            for pair in selected_ppi[gene]:
                if pair < gene:
                    continue
                G.edge(gene, pair, color='red', arrowhead='none')

        # TF edges
        if gene in selected_TF:
            for target in selected_TF[gene]:
                G.edge(gene, target, color='darkgreen', arrowhead='normal')

    G.render()
    return None




def plot_ppinetwork(number, results_dir, locus_activeset, network_ppi, phenotype_list, gene_phenoset, gene_special, degree_correction, TFsource_target):
    logger.info('plotting network')
    graph_name = 'ppi_network_pass' + str(number) +  '_degcor' + str(degree_correction)
    img_file = results_dir + graph_name + '.pdf'
    G = gv.Digraph(graph_name, filename=img_file,
                   engine='fdp', strict=True)  # 'neato' format='png'

    # PPI  pairs
    selected_ppi = dict()
    selected_genes = set()
    for locus in locus_activeset:
        selected_genes.update(locus_activeset[locus])

    for locus in locus_activeset:
        for gene in locus_activeset[locus]:
            if gene in network_ppi:
                paired = network_ppi[gene].intersection(selected_genes)
                if len(paired) > 0 :
                    selected_ppi[gene] = paired


    selected_TF = dict()
    for gene in selected_genes:
        if gene in TFsource_target:
            targets = TFsource_target[gene].intersection(selected_genes)
            selected_TF[gene] = targets

    # phenotype
    G.attr('node', shape='circle', fixedsize='true',
           width='0.3', fontsize='6')
    for phenotype in phenotype_list:
        G.node(phenotype)

    # ppi Genes - phenotype
    G.attr('node', fixedsize='true')
    for locus in locus_activeset:
        gene = list(locus_activeset[locus])[0]
        phenotypes = gene_phenoset[gene]
        special = gene_special[gene]
        color = 'white'
        if special:
            if 'omim' in special:
                color = 'red'
            elif 'lof' in special:
                color = 'orange'
            elif 'coloc' in special:
                color = 'yellow'



        if gene in selected_ppi:
            G.node(gene, style='filled', fillcolor=color, shape='rectangle', width='0.5',  height='0.2', fontsize='6')

            # gene-phenotype edges
            if phenotypes:
                pheno = list(phenotypes)[0].split('+')
                for ph in pheno:
                    G.edge(gene, ph, color='gray', arrowhead='none')

            # PPI edges
            for pair in selected_ppi[gene]:
                if pair < gene:
                    continue
                G.edge(gene, pair, color='red', arrowhead='none')

        # TF edges
        if gene in selected_TF:
            for target in selected_TF[gene]:
                G.edge(gene, target, color='darkgreen', arrowhead='normal')

    G.render()
    return None



def plot_ppitriangles(number, results_dir, locus_activeset, network_ppi, phenotype_list, gene_phenoset, gene_special, degree_correction):
    logger.info('plotting network')
    graph_name = 'triangle_network_pass' + str(number) +  '_degcor' + str(degree_correction)
    img_file = results_dir + graph_name + '.pdf'
    G = gv.Digraph(graph_name, filename=img_file,
                   engine='fdp', strict=True)  # 'neato' format='png'

    # PPI  pairs
    selected_ppi = dict()
    selected_genes = set()
    for locus in locus_activeset:
        selected_genes.update(locus_activeset[locus])

    for locus in locus_activeset:
        for gene in locus_activeset[locus]:
            if gene in network_ppi:
                paired = network_ppi[gene].intersection(selected_genes)
                if len(paired) > 0 :
                    selected_ppi[gene] = paired


    # phenotype
    G.attr('node', shape='circle', fixedsize='true',
           width='0.3', fontsize='6')
    for phenotype in phenotype_list:
        G.node(phenotype)

    for gene in selected_ppi:
        for neighbor1 in selected_ppi[gene]:
            for neighbor2 in selected_ppi[neighbor1]:
                #neighbor3_list = selected_ppi[neighbor2]
                if gene in selected_ppi[neighbor2]:
                    tri = [gene, neighbor1, neighbor2]


                    # ppi Genes - phenotype
                    G.attr('node', fixedsize='true')
                    for g in tri:
                        phenotypes = gene_phenoset[g]
                        special = gene_special[g]
                        color = 'white'
                        if special:
                            if 'omim' in special:
                                color = 'red'
                            elif 'lof' in special:
                                color = 'orange'
                            elif 'coloc' in special:
                                color = 'yellow'

                        G.node(g, style='filled', fillcolor=color, shape='rectangle', width='0.5',  height='0.2', fontsize='6')
                        # gene-phenotype edges
                        if phenotypes:
                            pheno = list(phenotypes)[0].split('+')
                            for ph in pheno:
                                G.edge(g, ph, color='gray', arrowhead='none')




                    # PPI edges
                    G.edge(g, neighbor1, color='red', arrowhead='none')
                    G.edge(g, neighbor2, color='red', arrowhead='none')
                    G.edge(neighbor2, neighbor1, color='red', arrowhead='none')



    G.render()
    return None



def plot_nchange(pass_locus_change, seed, ppi_run, degree_correction, TF_run, optim_run):
    # for each pass, number of loci that changed active genes
    nchange_series = []
    for p in pass_locus_change:
        nchange_series.append(len(pass_locus_change[p]))

    plt.plot(range(len(nchange_series)), nchange_series)
    plt.title('number of loci which changed as pass number')
    plt.savefig('./results/nchange_series_seed' + str(seed) + '_ppi' + str(ppi_run) +
                '_degcor' + str(degree_correction) + '_TF' + str(TF_run) + '_optim' + str(optim_run) + '.png')
    plt.close()


def plot_changed_loci(pass_locus_change, seed, ppi_run, degree_correction, TF_run, optim_run):
    # after burn-in, which loci changed genes and how often
    locus_nchange_series = dict()
    for p in range(len(pass_locus_change)):
        for locus, nchanges in pass_locus_change[p]:
            locus_nchange_series[locus] = nchanges

    loci_changed = []
    loci_nchange = []
    for locus in sorted(locus_nchange_series):
        loci_changed.append(locus)
        loci_nchange.append(locus_nchange_series[locus])

    plt.bar(loci_changed, loci_nchange)
    labels = [x[0:25] for x in loci_changed]
    plt.xticks(np.arange(len(labels)), labels,
               rotation=45, fontsize=8, ha='right')
    plt.title('nchanges of each locus, post burn-in')
    plt.tight_layout()
    plt.savefig('./results/loci_changed_seed' + str(seed) + '_ppi' + str(ppi_run) + '_degcor' + str(degree_correction) + '_TF' + str(TF_run) + '_optim' + str(optim_run) + '.png')
    plt.close() #clf()


def plot_convergence(maxdiff_list, seed, ppi_run, degree_correction, TF_run, optim_run):
    # # EVALUATION: convergence
    plt.plot(range(len(maxdiff_list)), maxdiff_list)
    plt.title('network covergence, post burn-in')
    plt.savefig('./results/convergence_seed' + str(seed) + '_ppi' + str(ppi_run) +
                '_degcor' + str(degree_correction) + '_TF' + str(TF_run) + '_optim' + str(optim_run) +'.png')
    plt.close()


def plot_networkscore(networkscore_list, locus_activeset, seed, ppi_run, degree_correction, TF_run, optim_run):
    # # EVALUATION: network total score
    nloci = len(locus_activeset)
    network_avg_score = (1 / nloci) * np.array(networkscore_list)
    plt.plot(network_avg_score)
    plt.title('average score of network')
    plt.savefig('./results/networkscore_seed' + str(seed) + '_ppi' + str(ppi_run) +
                '_degcor' + str(degree_correction) + '_TF' + str(TF_run) + '_optim' + str(optim_run) +'.png')
    plt.close()



def plot_scoregap(scores_logfile, seed, ppi_run, degree_correction, TF_run, optim_run):
    # # EVALUATION: which score (distance, special, ppi, TF) lead to selecting final gene
    # # coded in that order, i.e. (1000) means selected gene had best distance score in locus
    gap_logfile = './results/finalpass_gap_seed' + str(seed) + '_ppi' + str(ppi_run) +'_degcor' + str(degree_correction) + '_TF' + str(TF_run) + '_optim' + str(optim_run) +'.txt'
    fp = open(gap_logfile, 'w')
    fp.write('\t'.join(['locus', 'score_gap', 'candidate_gene', 'candidate_score', 'penulatimate_gene', 'penultimate_score']) + '\n')

    df = pd.read_csv(scores_logfile, delimiter='\t')
    locus_scoregap = []
    for locus in df['locus'].unique():
        tmp = df[df['locus'] == locus]
        locus_df_nrows = len(tmp) - 2
        locus_df = tmp.iloc[0:locus_df_nrows, :]
        locus_df.sort_values(by=['total_score'], ascending=False, inplace=True)
        if len(locus_df)<2:
            continue #locus_gap = 99
        locus_gap = locus_df.iloc[0,:]['total_score'] - locus_df.iloc[1,:]['total_score']
        locus_scoregap.append(locus_gap)

        locus_str = '\t'.join([locus, str(locus_gap),  locus_df.iloc[0,:]['gene'], str(locus_df.iloc[0,:]['total_score']), locus_df.iloc[1,:]['gene'] , str(locus_df.iloc[1,:]['total_score']) ])
        fp.write(locus_str + '\n')

    fp.close()


    plt.hist(locus_scoregap, bins=range(0, 30))
    plt.xticks(range(0, 30, 2))
    plt.title('Total-score difference between highest- \n and second highest ranked genes of each locus')
    plt.savefig('./results/gap_seed' + str(seed) + '_ppi' + str(ppi_run) +'_degcor' + str(degree_correction) + '_TF' + str(TF_run) + '_optim' + str(optim_run) + '.png')
    plt.close()





def plot_scorebreakdown(scores_logfile, seed, ppi_run, degree_correction, TF_run, optim_run):
    # # EVALUATION: which score (distance, special, ppi, TF) lead to selecting final gene
    # # coded in that order, i.e. (1000) means selected gene had best distance score in locus
    locus_scorebreakdown = []
    df = pd.read_csv(scores_logfile, delimiter='\t')
    for locus in df['locus'].unique():

        score_breakdown_str = ''
        tmp = df[df['locus'] == locus]
        # since we print two extra rows of information for each locus
        locus_df_nrows = len(tmp) - 2
        locus_df = tmp.iloc[0:locus_df_nrows, :]

        max_distance_score = np.max(locus_df['distance_score'])
        max_special_score = np.max(locus_df['special_score'])
        max_ppi_score = np.max(locus_df['ppi_score'])
        max_TF_score = np.max(locus_df['TF_score'])
        max_total_score = np.max(locus_df['total_score'])


        selected_gene = locus_df[locus_df['total_score'] == max_total_score]['gene'].values[0]  # NOTE WE HAVE ONE SELECTED GENE NOW
        selected_gene_distance_score = locus_df[locus_df['gene']
                                                == selected_gene]['distance_score'].values[0]
        selected_gene_special_score = locus_df[locus_df['gene']
                                               == selected_gene]['special_score'].values[0]

        selected_gene_ppi_score = locus_df[locus_df['gene']
                                           == selected_gene]['ppi_score'].values[0]

        selected_gene_TF_score = locus_df[locus_df['gene']
                                          == selected_gene]['TF_score'].values[0]


        special_score_locus = ''
        if (len(np.unique(locus_df['special_score'])) > 1) and (selected_gene_special_score == 0):
            special_score_locus = '0'
        if (selected_gene_special_score > 0):
            special_score_locus = '1'
        if (len(np.unique(locus_df['special_score'])) == 1) and (selected_gene_special_score == 0):
            special_score_locus = '_'

        ppi_score_locus = '_'
        if (len(np.unique(locus_df['ppi_score'])) > 1):
            ppi_score_locus =  int(selected_gene_ppi_score == max_ppi_score)

        TF_score_locus = '' #'_'
        if (len(np.unique(locus_df['TF_score'])) > 1):
           TF_score_locus =  int((selected_gene_TF_score == max_TF_score))

        score_breakdown = ['s', int(selected_gene_distance_score == max_distance_score), special_score_locus, ppi_score_locus, TF_score_locus]

        score_breakdown_str = ''.join([str(x) for x in score_breakdown])

        # write to df:
        df.loc[df['gene'] == selected_gene,
               'bestgene_breakdown'] = score_breakdown_str

    # write the new df:
    df.to_csv('./results/finalpass_scores_seed' + str(seed) +
              '_degcor' + str(degree_correction) + '_optim' + str(optim_run) + '.txt', sep='\t')

    m = ~df['bestgene_breakdown'].isnull()
    m2 = df[m]['bestgene_breakdown'].values

    df2 = df.groupby([df.bestgene_breakdown]).count()['locus']
    fig1 = plt.gcf()
    df2.plot(kind='bar')
    fig1.suptitle('categories: dist-special-ppi')
    fig1.savefig('./results/bestscore_breakdown_seed' + str(seed) + '_ppi' +
                 str(ppi_run) + '_degcor' + str(degree_correction) + '_TF' + str(TF_run) + '_optim' + str(optim_run) + '.png')
    plt.close()


def pathway_scores(network, pathway_dict, pathway_gene_universe, alpha):
    pathway_df = pd.DataFrame(columns=['pathway', 'p-value'])
    genes_in_network = get_networkgenes(network)
    for pathway in pathway_dict.keys():
        gene_in_both = pathway_dict[pathway].intersection(genes_in_network)
        count_11 = len(gene_in_both)
        if count_11 >= 5:
            #
            count_11 = len(gene_in_both)
            count_12 = len(genes_in_network) - count_11
            count_21 = len(pathway_dict[pathway]) - count_11
            count_22 = len(pathway_gene_universe) - (count_12 + count_21)
            table = np.array([[count_11, count_12], [count_21, count_22]])
            g, p, dof, expctd = chi2_contingency(table)
            if p < alpha:
                pathway_df = pathway_df.append(
                    {'pathway': pathway, 'p-value': p}, ignore_index=True)

    pathway_df.sort_values('p-value', inplace=True, ignore_index=True)
    return pathway_df


def get_networkgenes(network):
    ret = set()
    for locus in network:
        ret.update(network[locus])
    return (ret)

def get_specialcounts(network_dict, gene_special ):
    ##count how many genes in each special category:
    special_counts = dict()
    for locus in network_dict:
        for gene in network_dict[locus]:
           # pdb.set_trace()
            for special in gene_special[gene]:
               # pdb.set_trace()
                if special in special_counts:
                    special_counts[special] +=1
                else:
                    special_counts[special] = 0
    return special_counts

def write_shortdf(scores_logfile, seed, degree_correction, optim_run):
    df = pd.read_csv(scores_logfile, delimiter='\t')
    short_df = pd.DataFrame(columns = df.columns)
    for locus in df['locus'].unique():
        tmp = df[df['locus'] == locus]
        # since we print two extra rows of information for each locus
        locus_df_nrows = len(tmp) - 2
        locus_df = tmp.iloc[0:locus_df_nrows, :]
        max_total_score = np.max(locus_df['total_score'])
        selected_gene = locus_df[locus_df['total_score'] == max_total_score]['gene'].values[0]  # NOTE WE HAVE ONE SELECTED GENE NOW


        for i in range(len(locus_df)):
            row = locus_df.iloc[i, :]
            #print the genes that were selected but not min-distance, the min-dist gene(s), and the special genes
            if (row['gene'] == selected_gene) or (row['distance(kbp)'] == 0) or (not pd.isnull(row['special'])):
                                short_df = short_df.append(row, ignore_index=True)


    # write the new df:
    short_df.to_csv('./results/shortfinalpass_scores_seed' + str(seed) + '_degcor' + str(degree_correction) + '_optim' + str(optim_run) + '.txt', sep='\t')


def write_extended_finaldf(scores_logfile, seed, degree_correction, optim_run):
    df = pd.read_csv(scores_logfile, delimiter='\t')
    final_df = pd.DataFrame(columns = df.columns)
    final_df['mindist_gene']=False
    final_df['hiscore_gene'] = False
    final_df['selected_gene']= False
    final_df['hippi_gene']=False
    final_df['special_gene'] = False
    final_df['exp_score'] = 0

    for locus in df['locus'].unique():

        tmp = df[df['locus'] == locus]
        selected_gene = tmp.iloc[-2, :]['special'].split('\'')[1] #note that we only select one gene per locus


        # since we print two extra rows of information for each locus
        locus_df_nrows = len(tmp) - 2
        locus_df = tmp.iloc[0:locus_df_nrows, :]
        max_total_score = np.max(locus_df['total_score'])
        mindist = min(locus_df['distance(kbp)'])
        mindist_gene = locus_df[locus_df['distance(kbp)']==mindist]['gene'].values

        maxppi = max(locus_df['ppi_score'])
        ppi_diff = len(np.unique(locus_df['ppi_score']))



        scores = np.array(locus_df['total_score'])


        exp_scores = np.exp(scores-max_total_score)
        sum_scores = np.sum(exp_scores)
        exp_scores = exp_scores/sum_scores
        locus_df['exp_score'] = exp_scores


        for i in range(len(locus_df)):
            row = locus_df.iloc[i, :]
            if row['gene'] in mindist_gene:
                row['mindist_gene'] = True
            if row['total_score'] == max_total_score:
                row['hiscore_gene'] = True
            if row['gene'] == selected_gene:
                row['selected_gene'] = True
            if ((row['ppi_score'] == maxppi) and ppi_diff >1):
                row['hippi_gene'] = True
            if (not pd.isnull(row['special'])):
                row['special_gene'] = True
            final_df = final_df.append(row, ignore_index=True)

        # write the new df:
        final_df.to_csv('./results/extended_finalpass_scores_seed' + str(seed) + '_degcor' + str(degree_correction) + '_optim' + str(optim_run) + '.txt', sep='\t')
    return  final_df


def  write_inactive_special_df(extended_final_df, locus_activeset, locus_geneset, external_genes, seed, degree_correction, optim_run):
    selected_genes = set()
    for locus in locus_activeset:
        selected_genes.update(locus_activeset[locus])

    ###number of special genes not selected: 16 - from 12 loci
    inactive_specialgenes = sorted(external_genes - set(selected_genes))
    loci_inactive_specialgenes =[]
    for locus in locus_geneset:
        if len(locus_geneset[locus].intersection(inactive_specialgenes))>0:
            loci_inactive_specialgenes.append(locus)


    inactive_special_df = pd.DataFrame(columns = extended_final_df.columns)
    for locus in extended_final_df['locus'].unique():
        if locus not in loci_inactive_specialgenes:
            continue
        inactives_pecial_df = inactive_special_df.append(extended_final_df[extended_final_df['locus'] == locus], ignore_index=True)
    # write the new df:
    inactive_special_df.to_csv('./results/finalscores_inactive_special_seed' + str(seed) + '_degcor' + str(degree_correction) + '_optim' + str(optim_run) + '.txt', sep='\t')


def  write_inactive_ppi_df(extended_final_df, locus_activeset, locus_geneset, external_genes, seed, degree_correction, optim_run):
    inactive_ppi_df = pd.DataFrame(columns = extended_final_df.columns)
    for locus in extended_final_df['locus'].unique():
        bestbreakdowncol = extended_final_df[extended_final_df['locus']==locus]['bestgene_breakdown']
        if (bestbreakdowncol[~pd.isnull(bestbreakdowncol)].values[0][3] == '0'):
            inactive_ppi_df = inactive_ppi_df.append(extended_final_df[extended_final_df['locus'] == locus], ignore_index=True)
    # write the new df:
    inactive_ppi_df.to_csv('./results/finalscores_inactive_ppi_seed' + str(seed) + '_degcor' + str(degree_correction) + '_optim' + str(optim_run) + '.txt', sep='\t')



def write_neither_df(extended_final_df, locus_activeset, locus_geneset, external_genes, seed, degree_correction, optim_run):
    neither_df = pd.DataFrame(columns = extended_final_df.columns)
    for locus in extended_final_df['locus'].unique():
        bestbreakdowncol = extended_final_df[extended_final_df['locus']==locus]['bestgene_breakdown']
        if (bestbreakdowncol[~pd.isnull(bestbreakdowncol)].values[0] in ['s000','s0_0', 's0__', 's00_']):
            locus_df = extended_final_df[extended_final_df['locus'] == locus]
            for i in range(len(locus_df)):
                row = locus_df.iloc[i, :]
                expected_ppi = (not (locus_df.iloc[i, :]['potential_PPI']) == 0)
                special_g = ((locus_df.iloc[i, :]['special_gene']) == True)
                hippi_g = ((locus_df.iloc[i, :]['hippi_gene']) == True)
                dist_g = ((locus_df.iloc[i, :]['mindist_gene']) == True)
                best_g = ((locus_df.iloc[i, :]['selected_gene']) == True)

                if (expected_ppi or special_g or hippi_g or dist_g or best_g):
                    neither_df = neither_df.append(row, ignore_index=True)
    # write the new df:
    neither_df.to_csv('./results/finalscores_neither_seed' + str(seed) + '_degcor' + str(degree_correction) + '_optim' + str(optim_run) + '.txt', sep='\t')


def write_specialnotmindist_df(extended_final_df, locus_activeset, locus_geneset, external_genes, seed, degree_correction, optim_run):
    specialnotmindist_df = pd.DataFrame(columns = extended_final_df.columns)
    for locus in extended_final_df['locus'].unique():
        bestbreakdowncol = extended_final_df[extended_final_df['locus']==locus]['bestgene_breakdown']
        if (bestbreakdowncol[~pd.isnull(bestbreakdowncol)].values[0] in ['s010','s01_', 's011']):

            locus_df = extended_final_df[extended_final_df['locus'] == locus]
            for i in range(len(locus_df)):
                row = locus_df.iloc[i, :]
                #expected_ppi = (not (locus_df.iloc[i, :]['potential_PPI']) == 0)
                special_g = ((locus_df.iloc[i, :]['special_gene']) == True)
                #hippi_g = ((locus_df.iloc[i, :]['hippi_gene']) == True)
                dist_g = ((locus_df.iloc[i, :]['mindist_gene']) == True)
                best_g = ((locus_df.iloc[i, :]['selected_gene']) == True)

                if (special_g or dist_g or best_g):
                    specialnotmindist_df = specialnotmindist_df.append(row, ignore_index=True)
    # write the new df:
    specialnotmindist_df.to_csv('./results/finalscores_specialnotmindist_df_seed' + str(seed) + '_degcor' + str(degree_correction) + '_optim' + str(optim_run) + '.txt', sep='\t')



def write_ppinotmindist_df(extended_final_df, seed, degree_correction, optim_run):
    ppinotmindist_df = pd.DataFrame(columns = extended_final_df.columns)
    for locus in extended_final_df['locus'].unique():
        bestbreakdowncol = extended_final_df[extended_final_df['locus']==locus]['bestgene_breakdown']
        if (bestbreakdowncol[~pd.isnull(bestbreakdowncol)].values[0] in ['s001','s0_1']):

            locus_df = extended_final_df[extended_final_df['locus'] == locus]
            for i in range(len(locus_df)):
                row = locus_df.iloc[i, :]
                expected_ppi = (not (locus_df.iloc[i, :]['potential_PPI']) == 0)
                special_g = ((locus_df.iloc[i, :]['special_gene']) == True)
                hippi_g = ((locus_df.iloc[i, :]['hippi_gene']) == True)
                dist_g = ((locus_df.iloc[i, :]['mindist_gene']) == True)
                best_g = ((locus_df.iloc[i, :]['selected_gene']) == True)

                if (expected_ppi or special_g or hippi_g or dist_g or best_g):
                    ppinotmindist_df = ppinotmindist_df.append(row, ignore_index=True)
    # write the new df:
    ppinotmindist_df.to_csv('./results/finalscores_ppinotmindist_df_seed' + str(seed) + '_degcor' + str(degree_correction) + '_optim' + str(optim_run) + '.txt', sep='\t')


def plot_venn_selected(locus_activeset, seed, degree_correction, optim_run):

    file = './results/finalpass_scores_seed' + str(seed) + '_degcor' + str(degree_correction) + '_optim' + str(optim_run) + '.txt'
    df = pd.read_csv(file, delimiter='\t')



    s111 = sum(df['bestgene_breakdown']=='s111')
    selected_dist_ppi_special = s111

    s110 = sum(df['bestgene_breakdown']=='s110')
    s11_ = sum(df['bestgene_breakdown']=='s11_')
    selected_dist_special = s110+s11_

    s101 = sum(df['bestgene_breakdown']=='s101')
    s1_1 = sum(df['bestgene_breakdown']=='s1_1')
    selected_dist_ppi = s101+s1_1

    s011 = sum(df['bestgene_breakdown']=='s011')
    selected_special_ppi = s011

    s001 = sum(df['bestgene_breakdown']=='s001')
    s0_1 = sum(df['bestgene_breakdown']=='s0_1')
    selected_ppi = s001 + s0_1

    s010 = sum(df['bestgene_breakdown']=='s010')
    s01_ = sum(df['bestgene_breakdown']=='s01_')
    selected_special = s010 + s01_

    s0_0 = sum(df['bestgene_breakdown']=='s0_0')
    s0__ = sum(df['bestgene_breakdown']=='s0__')
    s00_ = sum(df['bestgene_breakdown']=='s00_')
    selected_neither = s0_0 + s0__ + s00_

    selected_dist = len(locus_activeset) - (selected_dist_ppi_special+selected_dist_special+selected_dist_ppi+selected_special_ppi+selected_ppi+selected_special+selected_neither)



    venn3(subsets = (selected_dist, selected_special, selected_dist_special , selected_ppi,selected_dist_ppi ,selected_special_ppi,selected_dist_ppi_special), set_labels = ('min_distance', 'special_gene', 'ppi'))

    fig1 = plt.gcf()
    plt.plot()
    fig1.suptitle('Selected Genes')
    fig1.savefig('./results/venn_selected_seed' + str(seed)  + '_degcor' + str(degree_correction) +  '_optim' + str(optim_run) + '.png')
    plt.close()


def plot_optimparams(distfac_active_list, distfac_inactive_list,  omim_list, lof_list, coloc_list, results_dir, seed):
    plt.plot(distfac_active_list , 'r')
    plt.plot(distfac_inactive_list, 'b')
    plt.title('Convergence of Distance Scale Parameters')
    plt.xlabel('Number of passes through network')
    plt.legend(["active", "inactive"])
    plt.grid()
    plt.savefig(results_dir + 'distance_scale_conv'+ str(seed)+'.pdf')
    plt.clf()



    plt.plot(omim_list, 'r')
    plt.plot(lof_list, 'b')
    plt.plot(coloc_list, 'g')
    plt.title('Convergence of Functional Score Weights')
    plt.xlabel('Number of passes through network')
    plt.legend(["omim", "exome", "coloc"])
    plt.grid()
    plt.savefig(results_dir + 'functional_weight_conv'+ str(seed)+'.pdf')
    plt.clf()



def run_experiment(phenotype_list, k, FLANKDIST, alpha, degree_correction, optim_run, network_plot, gene_PPIdegree, ppi_Ksquared, network_ppi, xnull, ppi_run, TF_run, TFsource_target, reverse_TF, TF_Ksquared, gene_TFindegree, gene_TFoutdegree, locus_geneset, networkgenes, gene_special, gene_distance, gene_signeddistance, external_genes, seed, results_dir, counts_log):


    random.seed(seed)
    print (seed)



    scores_logfile = os.path.join(
        results_dir, 'finalpass_scores_seed' + str(seed) + '_degcor' + str(degree_correction) + '_optim' + str(optim_run) +'.txt')
    pseleted_logfile = os.path.join(
        results_dir, 'pselected_seed' + str(seed) + '_degcor' + str(degree_correction) + '_optim' + str(optim_run) +'.txt')


    ##########################
    # Network Optimization
    ##########################



    universe_specialcounts = get_specialcounts(locus_geneset, gene_special )

    locus_activeset = initialize_activeset_random(locus_geneset, gene_distance)
    #locus_activeset = activeset_mindistance(locus_geneset, gene_distance)
    init_activeset = copy.deepcopy(locus_activeset)




    pass_locus_change = dict()
    locus_score = defaultdict(list)
    networkscore_list = []

    locus_nchange = defaultdict(dict)
    for locus in locus_activeset:
        locus_nchange[locus]['nchange'] = 0
        locus_nchange[locus]['changed'] = False

    locus_activelist_series = defaultdict(list)
    for locus in locus_activeset:
        locus_activelist_series[locus].append(list(locus_activeset[locus])[0])

    gene_nselected = dict()  # number of times selected
    gene_pselected = dict()  # probability selected, equal to ntimes / npass
    # initialize

    for locus in locus_geneset:
        for gene in locus_geneset[locus]:
            gene_nselected[gene] = 0  # integer number of times selected
            # number of times divided by number of passes but as floats!!!!!
            gene_pselected[gene] = 0.0

    npass = 100
    npass_completed = 0
    CONVERGENCE = 0.01


    MINPASS = 1 # do we need this anymore, now that we don't randomly initialize before each pass?!!!!
    maxdiff_list = []
    maxdiff_gene_list = []
    converged = False




    all_distances = []
    for locus in locus_geneset:
        for gene in locus_geneset[locus]:
            all_distances.append(gene_distance[gene])
    all_distances = np.array(all_distances)

    distfac_init = 0.5 * iqr(all_distances)




    params = {'omim': 10, 'lof': 8, 'coloc': 5, 'distfac_active': distfac_init, 'distfac_inactive': distfac_init}


    distfac_active_list = [params['distfac_active']]
    distfac_inactive_list = [params['distfac_inactive']]
    omim_list = [params['omim']]
    coloc_list = [params['coloc']]
    lof_list = [params['lof']]



    active_loc_guesses = []
    inactive_loc_guesses = []
    active_scale_guesses = []
    inactive_scale_guesses = []



    for pass_number in range(npass):

        onepass(locus_activeset, locus_activelist_series, locus_geneset, locus_score, locus_nchange, pass_locus_change,
                gene_distance, gene_special, params, network_ppi,  xnull, scores_logfile, gene_PPIdegree, degree_correction, ppi_Ksquared, TFsource_target, reverse_TF, TF_Ksquared, gene_TFindegree, gene_TFoutdegree)

        if optim_run:
            active_specialcounts = get_specialcounts(locus_activeset, gene_special )
            ngenes = len(get_networkgenes(locus_geneset))
            for special in active_specialcounts:
                n_activegenes = len(locus_activeset) #FOR NOW ONE GENE PER LOCUS
                n_11 = active_specialcounts[special]
                n_01 = n_activegenes - n_11 #active, but not feature
                n_10 = universe_specialcounts[special] - n_11#not active, but has feature
                n_00 = (ngenes - len(locus_activeset)) - (n_10)
                num = (n_11+1)/(n_01+1)
                den = (n_10+1)/(n_00+1)
                score = math.log(num/den)
                params[special] = score

            omim_list.append(params['omim'])
            coloc_list.append(params['coloc'])
            lof_list.append(params['lof'])


        active_distances = get_active_distance(locus_activeset, gene_distance) + 1
        inactive_distances = get_inactive_distance(locus_geneset, locus_activeset, gene_distance) + 1



        res = optimize.fsolve(distance_func, x0=1, args=(active_distances[:, None])) # The argument to the function is an array itself, so we need to introduce extra dimensions for dist.
        params['distfac_active'] = res[0]


        res = optimize.fsolve(distance_func, x0=1, args=(inactive_distances[:, None])) #
        params['distfac_inactive'] = res[0]


        distfac_active_list.append(params['distfac_active'])
        distfac_inactive_list.append(params['distfac_inactive'])


        active = active_distances
        inactive = inactive_distances

        #initialize
        if pass_number ==0 :
            both_colours, active_loc_init, active_scale_init, inactive_loc_init, inactive_scale_init = initialize(active, inactive)
            active_loc_guess = 0 #active_loc_init
            inactive_loc_guess = 0 #inactive_loc_init
            active_scale_guess =  active_scale_init # params['distfac_active'] #
            inactive_scale_guess = inactive_scale_init #params['distfac_inactive']  #

            active_scale_guesses.append(active_scale_guess)
            inactive_scale_guesses.append(inactive_scale_guess)




        if (network_plot and (not pass_number % 10)):
            plot_network(len(pass_locus_change), results_dir, locus_activeset,
                         network_ppi, phenotype_list, gene_phenoset, gene_special, degree_correction, TFsource_target)



        print(pass_number, 'pass finished, seed:', seed)
        assert (len(locus_score) == len(locus_geneset),
                "uh-oh locus_score has more/less loci than locus_geneset")

        # calculate network score of this pass
        networkscore = 0.0
        for locus in locus_score:
            networkscore += locus_score[locus][-1]
        networkscore_list.append(networkscore)

        for locus in locus_activeset:
            for g in locus_activeset[locus]:
                gene_nselected[g] = gene_nselected[g] + 1

        # maximum difference in pselected
        maxdiff = 0.0
        for g in gene_nselected:
            oldval = gene_pselected[g]
            newval = float(gene_nselected[g]) / float(pass_number + 1)
            gene_pselected[g] = newval
            diff = abs(oldval - newval)
            if (diff > maxdiff):
                maxdiff = diff
        maxdiff_list.append(maxdiff)


        current_pass = pass_number
        nchanges = len(pass_locus_change[current_pass])


        if (((maxdiff <= CONVERGENCE) and (pass_number >= MINPASS)) or ((nchanges == 0)   and (pass_number >= MINPASS))):
            logger.info('finished at pass %d', pass_number)
            converged = True
            if (network_plot and (pass_number % 10)):
                plot_network(len(pass_locus_change), results_dir, locus_activeset, network_ppi,
                             phenotype_list, gene_phenoset, gene_special, degree_correction, TFsource_target)

                plot_ppinetwork(len(pass_locus_change), results_dir, locus_activeset, network_ppi,
                             phenotype_list, gene_phenoset, gene_special, degree_correction, TFsource_target)

                plot_ppitriangles(len(pass_locus_change), results_dir, locus_activeset, network_ppi, phenotype_list, gene_phenoset, gene_special, degree_correction)

            break

    if not converged:
        logger.info('uh-oh! did not converge after %d passes', pass_number)




    # # EVALUATION plots ###########

    plot_nchange(pass_locus_change, seed,
                 ppi_run, degree_correction, TF_run, optim_run)

    plot_changed_loci(pass_locus_change, seed,
                      ppi_run, degree_correction, TF_run, optim_run)

    plot_convergence(maxdiff_list, seed, ppi_run, degree_correction, TF_run, optim_run)


    plot_scoregap(scores_logfile, seed, ppi_run, degree_correction, TF_run, optim_run)

    plot_networkscore(networkscore_list, locus_activeset,
                      seed, ppi_run, degree_correction, TF_run, optim_run)

    plot_scorebreakdown(scores_logfile, seed, ppi_run,  degree_correction, TF_run, optim_run)

    plot_venn_selected(locus_activeset, seed, degree_correction, optim_run)

    plot_optimparams(distfac_active_list, distfac_inactive_list,  omim_list, lof_list, coloc_list, results_dir, seed)




    # # EVALUATION ###########


    list_gene_counts = [] #[networkscore]
    for locus in sorted(locus_geneset.keys()):
        for gene in sorted(locus_geneset[locus]):
            gene_count = 0
            if gene in locus_activeset[locus]:
                gene_count = 1
            list_gene_counts.append(gene_count)



    df = pd.read_csv(counts_log, sep='\t')


    df[str(seed)] = list_gene_counts
    df.to_csv(counts_log, index=False, sep='\t')


    extendedfinal_df = write_extended_finaldf(scores_logfile, seed, degree_correction, optim_run)




def main():


    args = get_args()

    logger.info('args %s', str(args))
    phenotype_list = args.phenotype_list
    k = args.k_closestgenes
    FLANKDIST = args.flankdistance
    alpha = args.alpha
    degree_correction = not args.nodegcor
    optim_run = not args.nooptim
    seed_start = args.seed_start
    seed_end = args.seed_end


    network_plot = args.plot

    config = Config(args.config)

    logger.info('parameters: phenotypes %s k %d alpha %f',
                str(phenotype_list), k, alpha)


    ppi_file = config.ppi_file
    gene_file = config.gene_file
    protein_gene_file = config.protein_gene_file
    TopMedData_dir = config.TopMedData_dir
    GWAS_dir = config.GWAS_dir
    results_dir = config.results_dir
    omim_file = config.omim_file
    exome_file = config.exome_file


    isExist = os.path.exists(results_dir)
    if not isExist:
        os.makedirs(results_dir)


    # User cannot specify degree correction for PPI, if PPI files are not given
    if (ppi_file == 'None' or protein_gene_file == 'None'):
        degree_correction = False

    # Quick Data check - gene names matching?
    # file downloaded from ensembl, gene names
    gene_bp_dict = utils_data.read_gene_file(gene_file)
    ensembl_genes = set(gene_bp_dict.keys())

    # external genes:
    external_genesets = [utils_data.get_exome_geneset(exome_file), utils_data.get_coloc_geneset(
        TopMedData_dir, phenotype_list), utils_data.get_omim_geneset(omim_file)]

    external_genes = set()
    external_genes.update(*external_genesets)
    missing_genes = external_genes - ensembl_genes
    if missing_genes:
        logger.info(
            'uh-oh!!! data digging needed - gene(s) not found: %s', str(missing_genes))

    # RP11-325L7.1 is a QT exome gene with id: ENSG00000246323.2
    # Ensemble has ENSG00000246323.2 named as AC113382.1 in the downloaded file
    # here, we will rename RP11-325L7.1 to AC113382.1 for now doing it here, not changing file

    pheno_snp_geneset = defaultdict(dict)
    snp_gene_distance = defaultdict(dict)
    snp_gene_signeddist = defaultdict(dict)
    pheno_geneset = defaultdict(set)
    gene_phenoset = defaultdict(set)

    rep_snp = 0

    for ph in phenotype_list:
        pheno_snp_geneset[ph] = defaultdict(dict)
        pheno_snp_gene_distance = utils_data.get_network(
            k, FLANKDIST, TopMedData_dir, ph)


        for snp in pheno_snp_gene_distance:

            if snp in snp_gene_distance:
                rep_snp +=1

            # will rewrite repeat snps if >1 phen but ok
            snp_gene_distance[snp] = dict()
            genes, distances = list(zip(*pheno_snp_gene_distance[snp]))
            pheno_geneset[ph] = pheno_geneset[ph].union(set(genes))
            pheno_snp_geneset[ph][snp] = set(genes)

            for gene, distance in zip(genes, distances):
                snp_gene_distance[snp][gene] = abs(distance)
                snp_gene_signeddist[snp][gene] = distance
                gene_phenoset[gene] = gene_phenoset[gene].union([ph])

    # for network genes, get distance to closest snp and the closest snp
    gene_distance = get_gene_distance(snp_gene_distance)
    gene_signeddistance = get_gene_signeddistance(snp_gene_signeddist)



    snp_filename = os.path.join(results_dir, 'snps_to_genes.txt')
    write_snp_genes(snp_filename, snp_gene_distance, gene_phenoset)



    for gene in gene_distance:
        assert gene_distance[gene] == abs(gene_signeddistance[gene])


    gene_closestsnps = get_gene_closestsnps(snp_gene_distance, gene_distance)



    # for each network gene, indicate if and why it is special
    gene_special = utils_data.get_specialgenes(
        phenotype_list, TopMedData_dir, snp_gene_distance, exome_file, omim_file, gene_distance, delimiter='\t')

    # get phneotypes of omim genes
    omim_phenoset = utils_data.get_omim_pheno(omim_file)
    gene_phenoset.update(omim_phenoset)
    # clean any mismatches between ensembl and external gene name
    # update coloc gene name: RP11-325L7.1 to AC113382.1
    gene_special['AC113382.1'] = gene_special.pop('RP11-325L7.1')



    write_gene_closest(os.path.join(
        results_dir, 'gene_closest.txt'), gene_distance, gene_closestsnps)

    pheno_locus_gene_distance = defaultdict(dict)
    pheno_locus_geneset = defaultdict(dict)
    for ph in phenotype_list:
        pheno_snp_gene_distance = utils_data.get_network(
            k, FLANKDIST, TopMedData_dir, ph)
        locus_genes_distances = utils_data.merge_pheno_snps(
            pheno_snp_gene_distance)


        pheno_locus_gene_distance[ph] = defaultdict(dict)
        for locus in locus_genes_distances:
            genes, distances = list(zip(*locus_genes_distances[locus]))
            pheno_locus_geneset[ph][locus] = set(genes)
            pheno_locus_gene_distance[ph][locus] = defaultdict(dict)
            for gene, distance in zip(genes, distances):

                pheno_locus_gene_distance[ph][locus][gene] = distance



        filename = results_dir + ph + '_locus_gene_distance.txt'
        write_pheno_locus_genes(filename, ph,
                                locus_genes_distances)


    # keep track of gene-phenotype connections
    # concatenate the loci of all phenotypes
    union_locus_geneset = defaultdict(set)
    union_locus_gene_distance = defaultdict(dict)
    for ph in pheno_locus_geneset:
        union_locus_geneset.update(pheno_locus_geneset[ph])
        union_locus_gene_distance.update(pheno_locus_gene_distance[ph])


   # check that all special genes are in network
   # can miss if they are located > FLANKDIST from all SNPs and are not in the first k-genes of any SNP
    nw_missedgenes = set(gene_special.keys()) - set(gene_distance.keys())
    nw_missedgenes_special = [gene_special[x] for x in nw_missedgenes]


    #'DDX17' is coloc with greatest distance: 248139
    #closest SNP to 'KCNQ1' (exome) not in GWAS, from paper, this SNP is: rs2074238 - BUT THIS IS ALSO OMIM so it is added with its own locus





    for gene in nw_missedgenes:
        locus_name = 'locus_' + gene
        locus_loc = int((int(gene_bp_dict[gene]['gene_tss'])+int(gene_bp_dict[gene]['gene_end']))/2 - int(gene_bp_dict[gene]['gene_tss']))
        gene_distance[gene] = locus_loc
        gene_signeddistance[gene] = locus_loc
        gene_closestsnps[gene] = locus_name
        snp_gene_distance[locus_name][gene] = locus_loc
        snp_gene_signeddist[locus_name][gene] = locus_loc
        union_locus_geneset[locus_name] = {gene}
        union_locus_gene_distance[locus_name][gene] = locus_loc


    assert(not(set(gene_special.keys()) - set(gene_distance.keys())))


    locus_to_genes = utils_data.merge_loci(
        union_locus_geneset, union_locus_gene_distance)


    filename = os.path.join(results_dir, 'locus_to_genes.txt')
    locus_gene_distance = dict()
    locus_geneset = dict()
    for locus in locus_to_genes:
        locus_gene_distance[locus] = dict()
        locus_geneset[locus] = set()
        for (g, d) in locus_to_genes[locus]:
            locus_gene_distance[locus][g] = d
            locus_geneset[locus].add(g)

    # check locus width
    snp_bp_dict = defaultdict(dict)
    for phenotype in phenotype_list:
        SNPfile = TopMedData_dir + phenotype + '_rsid_GRCh38p13.txt'
        ph_snp_bp = utils_data. read_snp_file(SNPfile)
        snp_bp_dict.update(ph_snp_bp)

    loci_name_list = []
    loci_width_list = []
    loci_nsnp_list = []
    loci_ngene_list = []
    for locus in locus_geneset:
        w = 0
        if '+' in locus:
            w+=1
            snp_list = locus.split('+')
            bp_list = []
            for snp in snp_list:
                snp_bp = int(snp_bp_dict[snp]['rsid_start'])
                bp_list.append(snp_bp)


            snp_bp = tuple(zip(snp_list, bp_list))
            snp_bp = sorted(snp_bp, key=lambda x: x[1])

            locus_width = (snp_bp[-1][1]- snp_bp[0][1])
            locus_nsnp = len(snp_list)
            locus_ngene = len(locus_geneset[locus])

            loci_name_list.append(locus)
            loci_width_list.append(locus_width)
            loci_nsnp_list.append(locus_nsnp)
            loci_ngene_list.append(locus_ngene)


    plt.scatter(loci_width_list, loci_nsnp_list)
    plt.title('number of SNPs in each locus')
    plt.xlabel('locus width')
    plt.grid()
    plt.savefig(results_dir + 'loci_nsnp_vs_width.png')
    plt.clf()



    plt.scatter(loci_width_list, loci_ngene_list)
    plt.title('number of genes in each locus')
    plt.xlabel('locus width')
    plt.grid()
    plt.savefig(results_dir + 'loci_gene_vs_width.png')
    plt.clf()



    write_locus_genes(filename, locus_gene_distance, gene_phenoset)


    networkgenes = get_networkgenes(locus_geneset)

    logger.info('network has %d SNPs, %d loci, and %d genes', len(
        snp_gene_distance), len(locus_geneset), len(networkgenes))

    # bar plot ngenes per loci:
    locus_ngenes = dict()
    for locus in locus_geneset:
        locus_ngenes[locus] = len(locus_geneset[locus])

    loci_list = locus_ngenes.keys()
    ngenes_list = locus_ngenes.values()
    plt.bar(loci_list, ngenes_list)
    plt.title('ngenes of loci')
    plt.tight_layout()
    plt.savefig('./results/loci_ngenes.png')
    plt.clf()


    plt.hist(ngenes_list, bins=200)
    plt.title('ngenes of loci')
    plt.tight_layout()
    plt.savefig('./results/loci_ngenes_hist.png')
    plt.clf()


    nsnps_ngenes = []
    for x in locus_ngenes:
        nsnps_ngenes.append((locus_ngenes[x], len(x.split('+'))))
    nsnps, ngenes = list(zip(*nsnps_ngenes))
    nsnps = list(nsnps)
    ngenes = list(ngenes)
    plt.scatter(nsnps, ngenes, marker='o')
    plt.title('number of SNPs in locus vs number of genes')
    plt.tight_layout()
    plt.savefig('./results/ngenes_snp_scatter.png')
    plt.clf()



    network_ppi = None
    xnull = None
    gene_PPIdegree = None
    ppi_Ksquared = None
    ppi_run = False
    TF_run = False

    if (os.path.exists(protein_gene_file) and os.path.exists(ppi_file)):
        ppi_run = True
        # implement PPI interactions in the Lk network
        # fnull is the null hypothesis for the probability that a pair of proteins has an interaction
        # do we just count interactions between loci? or also within loci?
        # might as well count them all and divide by number of pairs
        uniprot_to_hgnc = utils_ppi.protein_gene(protein_gene_file)
        uniprot_directed_dict = utils_ppi.read_ppi(ppi_file)
        protein_directed_dict = utils_ppi.convert_uniprot_to_hgnc(
            uniprot_directed_dict, uniprot_to_hgnc)
        protein_proteins = utils_ppi.make_undirected(protein_directed_dict)
        network_ppi = dict()
        for p in protein_proteins:
            if p in networkgenes:
                network_ppi[p] = protein_proteins[p].intersection(
                    networkgenes)
                network_ppi[p].discard(p)
                if len(network_ppi[p]) == 0:
                    del network_ppi[p]



        max_networkedges = utils_ppi.count_undirected_within_set(
            networkgenes, network_ppi)
        logger.info('%d max possible edges in network', max_networkedges)

        # no degree_correction:
        fnull = utils_ppi.get_fnull(networkgenes, network_ppi)
        v = float(len(locus_geneset))  # pick only one gene per locus
        xnull = 0.5 * v * (v - 1) * fnull

        nlocus = len(locus_geneset)
        logger.info('%d loci', nlocus)
        logger.info('%f degree (null)', fnull * (nlocus - 1.0))
        logger.info('%f edges (null)', 0.5 * fnull * nlocus * (nlocus - 1.0))

        # degree_correction:
        gene_PPIdegree = utils_ppi.get_degree(networkgenes, network_ppi)

        Dmax_squared = 0
        for gene in gene_PPIdegree:
            Dmax_squared += gene_PPIdegree[gene]**2
        ppi_Ksquared = 2 * max_networkedges * \
            (1 - (Dmax_squared / (4 * max_networkedges**2)))  # K**2

        D1max, D2max = utils_ppi.get_D_other(networkgenes, gene_PPIdegree)
        xnull_max = (1 / (2 * ppi_Ksquared)) * \
            ((D1max)**2 - (D2max))  # ==nedge_max

    TF_Ksquared = None
    TFsource_target = None
    reverse_TF = None
    gene_TFoutdegree = None
    gene_TFindegree = None

    counts_log = './results/counts.txt'
    fp = open(counts_log, 'w')
    fp.write('\t'.join(['', 'seed:'])+ '\n')
    fp.write('\t'.join(['', 'network_score:'])+ '\n')
    for locus in sorted(locus_geneset.keys()):
        for gene in sorted(locus_geneset[locus]):
            fp.write('\t'.join([locus, gene]) + '\n')
    fp.close()


    locus_nppigene = dict()
    for locus in locus_geneset:
        locus_nppigene[locus] = 0
        for gene in locus_geneset[locus]:
            if gene in network_ppi:
                locus_nppigene[locus]+=1

    with open('./results/locus_nppi.txt', 'w') as f:
        f.write('\t'.join(['locus', 'nppi']) + '\n')
        for locus in locus_nppigene:
            f.write('\t'.join([locus, str(locus_nppigene[locus])]) + '\n')


    with open('./results/gene_nppi.txt', 'w') as f:
        f.write('\t'.join(['gene', 'PPIdegree']) + '\n')
        for locus in sorted(locus_geneset.keys()):
            for gene in sorted(locus_geneset[locus]):
                f.write('\t'.join([gene, str(gene_PPIdegree[gene])]) + '\n')

    print ('FINISHED NETWORK CONSTRUCTION')



    init_countsdf = pd.DataFrame()
    network_locus_list = []
    network_gene_list = []
    for locus in sorted(locus_geneset.keys()):
        for gene in sorted(locus_geneset[locus]):
            network_locus_list.append(locus)
            network_gene_list.append(gene)
    init_countsdf['locus'] = network_locus_list
    init_countsdf['gene'] = network_gene_list
    allruns_countlog = './results/counts.txt'
    init_countsdf.to_csv(allruns_countlog, index=False, sep='\t')



    for seed in range(seed_start, seed_end):
        print (seed)
        run_experiment(phenotype_list, k, FLANKDIST, alpha, degree_correction, optim_run, network_plot, gene_PPIdegree, ppi_Ksquared, network_ppi, xnull, ppi_run, TF_run, TFsource_target, reverse_TF, TF_Ksquared, gene_TFindegree, gene_TFoutdegree, locus_geneset, networkgenes, gene_special, gene_distance, gene_signeddistance, external_genes, seed, results_dir, counts_log)


        counts_df = pd.read_csv(allruns_countlog, sep='\t')
        file = './results/extended_finalpass_scores_seed' + str(seed) + '_degcorTrue_optimTrue.txt'
        pass_df = pd.read_csv(file, sep='\t')
        loci = counts_df['locus']


        gene_selection_list = []
        for locus in np.unique(loci):
            locus_df = pass_df[pass_df['locus'] == locus]
            selectedgene_row = ~locus_df['bestgene_breakdown'].isna()
            gene_selection = list(selectedgene_row.astype(int).values)
            gene_selection_list.extend(gene_selection)


        counts_df[str(seed)] = gene_selection_list
        counts_df.to_csv(allruns_countlog, index=False, sep='\t')





if __name__ == "__main__":
    main()

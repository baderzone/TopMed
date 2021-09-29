

import argparse
import yaml
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.stats import sem
import os as os
import csv
import pdb
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


def get_args():
    parser = argparse.ArgumentParser(description='Select genes')
    parser.add_argument('-s0', '--seed_start', help='seed0',
                        required=False, type=int, default=0)
    parser.add_argument('-s1', '--seed_end', help='seed1',
                        required=False, type=int, default=1)
    parser.add_argument('-p', '--prob_threshold',
                        help='prob_threshold', required=False, type=float, default=1)

    args = parser.parse_args()
    return(args)


args = get_args()
seed_start = args.seed_start
seed_end = args.seed_end
prob_threshold = args.prob_threshold


# read counts file, find probabilities
allruns_countlog = './results/counts.txt'
# Now create summary df
counts_df = pd.read_csv(allruns_countlog, sep='\t')
# number times the gene was selected
nselected = counts_df.iloc[:, 2:].sum(axis=1)
nseed = len(counts_df.columns) - 2  # number of 'other' columns
counts_df['prob_selected'] = (nselected / nseed)
counts_df.to_csv('./results/counts_updated.txt', index=False, sep='\t')


genestats_file = './results/extended_finalpass_scores_seed0_degcorTrue_optimTrue.txt'
genestats_df = pd.read_csv(genestats_file, sep='\t')


# DECISIVE (prob selected > prob_threshold)
decisive_df = counts_df.loc[(counts_df['prob_selected'] == 0) | (
    counts_df['prob_selected'] >= prob_threshold)]
decisive_genes = decisive_df.iloc[:, 1]

decisive_loci = decisive_df['locus'].unique()
decisive_df.to_csv('./results/decisive_log.txt', index=False, sep='\t')

genestats_file = './results/extended_finalpass_scores_seed0_degcorTrue_optimTrue.txt'
genestats_df = pd.read_csv(genestats_file, sep='\t')
decisivestats_df = genestats_df[genestats_df['locus'].isin(decisive_loci)]
decisivestats_df = decisivestats_df[[
    'locus', 'gene', 'bestgene_breakdown', 'mindist_gene', 'distance(kbp)', 'special']]

decisivestats_df.to_csv('./results/decisive.txt', index=False, sep='\t')
tmpZ = decisive_df[['locus', 'gene', 'prob_selected']]
decisivestats_df = pd.merge(tmpZ, decisivestats_df, on=[
                            "locus", "gene"], validate='one_to_one')
decisivestats_df.to_csv(
    './results/decisiveloci_countsummary_geneestats.txt', index=False, sep='\t')


decisive_onlydf = decisivestats_df[decisivestats_df['prob_selected']
                                   >= prob_threshold]
decisive_onlydf.to_csv(
    './results/decisive_summarystats.txt', index=False, sep='\t')


s111 = sum(decisive_onlydf['bestgene_breakdown'] == 's111')
selected_dist_ppi_special = s111

s110 = sum(decisive_onlydf['bestgene_breakdown'] == 's110')
s11_ = sum(decisive_onlydf['bestgene_breakdown'] == 's11_')
selected_dist_special = s110 + s11_

s101 = sum(decisive_onlydf['bestgene_breakdown'] == 's101')
s1_1 = sum(decisive_onlydf['bestgene_breakdown'] == 's1_1')
selected_dist_ppi = s101 + s1_1

s011 = sum(decisive_onlydf['bestgene_breakdown'] == 's011')
selected_special_ppi = s011

s001 = sum(decisive_onlydf['bestgene_breakdown'] == 's001')
s0_1 = sum(decisive_onlydf['bestgene_breakdown'] == 's0_1')
selected_ppi = s001 + s0_1

s010 = sum(decisive_onlydf['bestgene_breakdown'] == 's010')
s01_ = sum(decisive_onlydf['bestgene_breakdown'] == 's01_')
selected_special = s010 + s01_

s0_0 = sum(decisive_onlydf['bestgene_breakdown'] == 's0_0')
s0__ = sum(decisive_onlydf['bestgene_breakdown'] == 's0__')
s00_ = sum(decisive_onlydf['bestgene_breakdown'] == 's00_')
selected_neither = s0_0 + s0__ + s00_

selected_dist = len(decisive_onlydf) - (selected_dist_ppi_special + selected_dist_special +
                                        selected_dist_ppi + selected_special_ppi + selected_ppi + selected_special + selected_neither)


venn3(subsets=(selected_dist, selected_special, selected_dist_special, selected_ppi, selected_dist_ppi,
               selected_special_ppi, selected_dist_ppi_special), set_labels=('min_distance', 'special_gene', 'ppi'))

fig1 = plt.gcf()
plt.plot()
fig1.suptitle('Strongly Selected Genes over 100 runs ')
fig1.savefig('./results/venn_decisive.png')
plt.close()


# INDECISIVE (prob selected between 0 and prob_threshold)
tmp = counts_df.loc[counts_df['prob_selected'] > 0]
indecisive_df = tmp.loc[tmp['prob_selected'] < prob_threshold]
indecisive_genes = indecisive_df['gene'].values


ret = defaultdict(lambda: defaultdict(list))
for seed in range(seed_start, seed_end):
    file = './results/extended_finalpass_scores_seed' + \
        str(seed) + '_degcorTrue_optimTrue.txt'

    with open(file) as ifile:
        reader = csv.DictReader(ifile, delimiter='\t')
        for row in reader:

            gene = row['gene']

            if gene not in set(indecisive_genes):
                continue

            exp_score = row['exp_score']
            total_score = row['total_score']
            distance_score = row['distance_score']
            ppi_score = row['ppi_score']
            special_score = row['special_score']

            ret[gene]['total_score'].append(total_score)
            ret[gene]['exp_score'].append(exp_score)
            ret[gene]['distance_score'].append(distance_score)
            ret[gene]['ppi_score'].append(ppi_score)
            ret[gene]['special_score'].append(special_score)

            # repeat, but whatever:
            distance = row['distance(kbp)']
            special = row['special']
            locus = row['locus']
            potential_PPI = row['potential_PPI']
            ret[gene]['distance'] = distance
            ret[gene]['special'] = special
            ret[gene]['locus'] = locus
            ret[gene]['potential_PPI'] = potential_PPI
            ret[gene]['probselected'] = indecisive_df.loc[indecisive_df['gene']
                                                          == gene, 'prob_selected'].values[0]

for gene in ret.keys():
    ret[gene]['avg_totalscore'] = np.mean(
        [float(x) for x in (np.array(ret[gene]['total_score']))])
    ret[gene]['avg_expscore'] = np.mean(
        [float(x) for x in (np.array(ret[gene]['exp_score']))])
    ret[gene]['sem_expscore'] = sem(
        [float(x) for x in (np.array(ret[gene]['exp_score']))])
    ret[gene]['avg_distancescore'] = np.mean(
        [float(x) for x in (np.array(ret[gene]['distance_score']))])
    ret[gene]['avg_ppiscore'] = np.mean(
        [float(x) for x in (np.array(ret[gene]['ppi_score']))])
    ret[gene]['avg_specialscore'] = np.mean(
        [float(x) for x in (np.array(ret[gene]['special_score']))])

new_file = './results/aggregated_indecisive_EXP.txt'
fp = open(new_file, 'w')
fp.write('\t'.join(['locus', 'gene', 'prob_selected', 'avg_totalscore', 'avg_expscore', 'sem_avgscore', 'avg_ppiscore',
                    'avg_distancescore', 'distance', 'avg_specialscore', 'special']) + '\n')

for gene in ret.keys():

    gene_str = '\t'.join([ret[gene]['locus'], gene, str(ret[gene]['probselected']), str(ret[gene]['avg_totalscore']), str(ret[gene]['avg_expscore']), str(ret[gene]['sem_expscore']), str(ret[gene]['avg_ppiscore']),
                          str(ret[gene]['avg_distancescore']), str(ret[gene]['distance']), str(ret[gene]['avg_specialscore']), str(ret[gene]['special'])])

    fp.write(gene_str + '\n')
fp.close()

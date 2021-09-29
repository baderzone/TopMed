#!/usr/bin/env python
# coding: utf-8


import numpy as np
import csv
from collections import defaultdict
import warnings
from collections import Counter
from operator import itemgetter
import pandas as pd
import os
import pdb

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_gene_file(file, annotated_only=False, delimiter='\t'):
    skipped = []

    ret = defaultdict(dict)

    with open(file) as fp:

        # header = [s.replace(" ", "") for s in fp.readline().split('\t')]
        #reader = csv.DictReader(fp, fieldnames=header)

        reader = csv.DictReader(fp, delimiter='\t')
        #header = next(reader)

        m = 0
        for row in reader:

            (gene, chromosome, gene_tss, gene_end, gene_type, gene_description, gene_phenotype) = (
                row['Gene name'], row['Chromosome/scaffold name'], row['Transcription start site (TSS)'], row['Gene end (bp)'], row['Gene type'], row['Gene description'], row['Phenotype description'])

            if chromosome.startswith('CHR'):
                continue

            if annotated_only == True:
                if re.search(r'\A[A][a-zA-Z]\d{6}', gene):
                    skipped.append(gene)
                    continue

            if gene == 'TRDN':
                m += 1

            if (gene == 'TRDN' and m > 1):
                continue  # last row entry has tss and end site the same

            ret[gene]['chromosome'] = chromosome
            ret[gene]['gene_tss'] = gene_tss
            ret[gene]['gene_end'] = gene_end
            ret[gene]['gene_type'] = gene_type
            ret[gene]['gene_description'] = gene_description
            if 'gene_phenotype' not in ret[gene]:
                ret[gene]['gene_phenotype'] = set()
            ret[gene]['gene_phenotype'].add(gene_phenotype)

    fp.close()

    n_skipped = len(np.unique(skipped))

    logger.info('%d skipped (unannotated) genes', n_skipped)

    # m = pd.DataFrame.from_dict(ret, orient='index')
    # m = m[['chromosome', 'gene_tss', 'gene_end']]
    # m.to_csv('GeneType_GRCh38p13_condensed.txt', sep='\t')

    return(ret)


def read_gene_file_old(file, delimiter='\t'):
    fp = open(file, 'r')
    ret = defaultdict(dict)

    next(fp)  # this skips the first line (header line)

    for line in fp:

        toks = line.strip().split(delimiter)
        (gene, chromosome, gene_type, gene_start, gene_end) = (
            toks[5], toks[2], toks[3], toks[0], toks[1])

        if chromosome.startswith('CHR'):
            continue

        ret[gene]['chromosome'] = chromosome
        ret[gene]['gene_type'] = gene_type
        ret[gene]['gene_start'] = gene_start
        ret[gene]['gene_end'] = gene_end

    fp.close()
    return(ret)


def read_snp_file(file, delimiter='\t'):

    fp = open(file, 'r')
    ret = defaultdict(dict)

    next(fp)  # this skips the first line (header line)

    for line in fp:
        toks = line.strip().split(delimiter)
        # print(length(toks))
        (rsid, chromosome, rsid_start, rsid_end) = (
            toks[0], toks[2], toks[3], toks[4])

        if chromosome.startswith('CHR'):
            continue

        if rsid in ret:
            warnings.warn('repeat SNP %s' % (rsid))

        ret[rsid]['chromosome'] = chromosome
        ret[rsid]['rsid_start'] = rsid_start
        ret[rsid]['rsid_end'] = rsid_end

    fp.close()
    return(ret)


def read_gwas_file(file, delimiter='\t'):
    ret = dict()

    with open(file) as ifile:
        reader = csv.DictReader(ifile, delimiter=delimiter)
        for row in reader:
            rsid = row['SNPS']
            if rsid not in ret:
                ret[rsid] = defaultdict(list)
                # print(rsid)
                ret[rsid]['pubmed'] = row['PUBMEDID']
                ret[rsid]['type'] = row['CONTEXT']
                # NOTE: in the current version I don't consider p-value
                ret[rsid]['p_value'].append(row['P-VALUE'])
            else:
                ret[rsid]['p_value'].append(row['P-VALUE'])

    return(ret)


def get_snp_gene_distance(gene_bp_dict, snp_bp_dict):
   # calculate distance to all genes on same chromosome
    ret = defaultdict(list)
    mindist_genes_dict = defaultdict(set)

    for variant in snp_bp_dict.keys():

        variant_list = []

        # find position of variant
        variant_chromosome = snp_bp_dict[variant]['chromosome']
        variant_bp = int(snp_bp_dict[variant]['rsid_start'])
        # find distance to all genes in that chromosome:
        for gene in gene_bp_dict.keys():
            if not gene_bp_dict[gene]['chromosome'] == variant_chromosome:
                continue
            else:
                gene_tss = int(gene_bp_dict[gene]['gene_tss'])
                gene_end = int(gene_bp_dict[gene]['gene_end'])

                distance = (gene_tss - variant_bp)

            variant_list.append((gene, distance))

        ret[variant] = sorted(variant_list, key=lambda x: abs(x[1]))

    return (ret)

def get_network(k, FLANKDIST, TopMedData_dir, phenotype):

    # SNP file is the SNP information from ensembl
    #SNPfile = TopMedData_dir + phenotype + '_GWAScat_rsid_GRCh38p13.txt'
    SNPfile = TopMedData_dir + phenotype + '_rsid_GRCh38p13.txt'

    snp_bp_dict = read_snp_file(SNPfile)

    # Rows from the GWAS catalog
    # gwas_file = TopMedData_dir + phenotype + '_GWAScat.txt'
    # gwas_dict = read_gwas_file(gwas_file)

    # Genes information from ensembl
    gene_file = TopMedData_dir + 'GeneType_GRCh38p13.txt'
    gene_bp_df = pd.read_csv(gene_file, sep='\t')
    gene_bp_df = gene_bp_df.set_index('gene')
    gene_bp_dict = gene_bp_df.to_dict(orient="index")

    # for each variant, a list of all genes in the chromosome sorted by distance
    phen_dict = get_snp_gene_distance(gene_bp_dict, snp_bp_dict)

    # limit to the closest k genes
    #phen_dict[variant] = phen_dict[variant][0:k]

    for variant in phen_dict.keys():
        # closest k genes
        variant_genedist = phen_dict[variant][0:k]
        # add more genes based on distance of kth gene
        for gene, distance in phen_dict[variant][k:]:

            if abs(distance) < FLANKDIST:
                variant_genedist.append((gene, distance))
            else:
                break
        phen_dict[variant] = variant_genedist

    # get unique genes
    uniq_genes = dict()
    for v in phen_dict:
        for g in phen_dict[v]:
            uniq_genes[g] = uniq_genes.get(g, 0) + 1
    gene_hist = dict()
    for g in uniq_genes:
        cnt = uniq_genes[g]
        gene_hist[cnt] = gene_hist.get(cnt, 0) + 1

    print('phenotype %s\tvariants %d\tgenes %d\thistogram %s' %
          (phenotype, len(phen_dict), len(uniq_genes), str(gene_hist)))

    return phen_dict


def get_coloc_geneset(TopMedData_dir, phenotype_list):
    ret = set()
    for phenotype in phenotype_list:
        GTEx_inputfile = TopMedData_dir + phenotype + '_GTEx.txt'
        if not os.path.exists(GTEx_inputfile):
            logger.info('coloc data for pheno %s does not exist', phenotype)
            continue
        GTEx_df = pd.read_csv(GTEx_inputfile, delimiter=',')
        for index, row in GTEx_df.iterrows():
            coloc_gene = row['Colocalizing_gene']
            ret.add(coloc_gene)
    return (ret)


def get_exome_geneset(exome_file):
    print(exome_file)
    fp = open(exome_file, 'r', encoding="utf16")
    ret = set()
    for line in fp:
        gene = line.strip()
        ret.add(gene)
    return (ret)


def get_missense_geneset(TopMedData_dir, phenotype_list, snp_gene_distance, delimiter='\t'):
    ret = set()
    # GWAS
    for phenotype in phenotype_list:

        file = TopMedData_dir + phenotype + '_GWAScat.txt'
        with open(file) as ifile:
            reader = csv.DictReader(ifile, delimiter=delimiter)
            for row in reader:
                snp = row['SNPS']
                function = row['CONTEXT']
                # for now only protein coding snp type is 'missense_variant'
                if (function == 'missense_variant'):
                    for gene in snp_gene_distance[snp]:
                        d = snp_gene_distance[snp][gene]
                        if d == 0:
                            ret.add(gene)
    return ret


def get_omim_geneset(mendelian_file):
    fp = open(mendelian_file, 'r', encoding="utf16")
    next(fp)  # this skips the first line (header line)
    ret = set()
    for line in fp:
        gene = line.strip().split('\t')[2]
        ret.add(gene)
    return (ret)


def get_omim_pheno(mendelian_file):
    fp = open(mendelian_file, 'r', encoding="utf16")
    next(fp)  # this skips the first line (header line)
    ret = defaultdict(set)
    for line in fp:
        toks = line.strip().split('\t')
        gene, pheno = toks[2], toks[1]
        ret[gene].add(pheno)
    return (ret)


def get_specialgenes(phenotype_list, TopMedData_dir, snp_gene_distance, exome_file, omim_file, gene_distance, delimiter='\t'):
    # coding_snps (missense for now)
    gene_types = dict()
    missense_geneset = get_missense_geneset(
        TopMedData_dir, phenotype_list, snp_gene_distance, delimiter='\t')
    for gene in missense_geneset:
        gene_types[gene] = {'lof'}
    # colocalized
    coloc_geneset = get_coloc_geneset(TopMedData_dir, phenotype_list)
    for gene in coloc_geneset:
        if gene in gene_types:
            gene_types[gene].add('coloc')
        else:
            gene_types[gene] = {'coloc'}
    # Exome
    exome_geneset = get_exome_geneset(exome_file)
    for gene in exome_geneset:
        if gene in gene_types:
            gene_types[gene].add('lof')
        else:
            gene_types[gene] = {'lof'}

    # Mendelian
    omim_geneset = get_omim_geneset(omim_file)
    for gene in omim_geneset:
        if gene in gene_types:
            gene_types[gene].add('omim')
        else:
            gene_types[gene] = {'omim'}

    for gene in gene_distance:
        if gene not in gene_types:
            gene_types[gene] = {}

    return(gene_types)


def merge_pheno_snps(snp_to_genes):
    # snp_to_genes is a dictionary, key = snp, value = list of genes
    # output should be similar, but merging SNPs whose genes overlap
    # the important thing is that if v is a value, then we can build a dictionary seen(v)
    # and i don't know if that will work if v is a tuple
    # but also if a gene is assigned to multiple variants, the distances could be different
    # so we really just want the gene names
    locus_to_genes = defaultdict(list)  # genes as a list
    gene_to_locus = dict()  # gene points to a single locus
    for snp in snp_to_genes:
        # i think this is the same as genes, distances = zip(*snp_to_genes) but i'm never sure
        genes = [x[0] for x in snp_to_genes[snp]]
        distances = [x[1] for x in snp_to_genes[snp]]
        # figure out if any of the genes is already in a locus
        merge = dict()
        for g in genes:
            if g in gene_to_locus:
                merge[gene_to_locus[g][0]] = True

        if not merge:
            for i in range(len(genes)):
                g = genes[i]
                d = abs(distances[i])
                gene_to_locus[g] = (snp, d)
                locus_to_genes[snp].append((g, d))
        else:
            all_genes_dist = dict()
            for i in range(len(genes)):
                g = genes[i]
                d = abs(distances[i])
                all_genes_dist[g] = d

            toks = [snp]
            for m in merge:
                toks = toks + m.split('+')
                other_snp_genes = [x[0] for x in locus_to_genes[m]]
                for g in other_snp_genes:
                    d = gene_to_locus[g][1]
                    if g in all_genes_dist:
                        d = min(all_genes_dist[g], d)
                    all_genes_dist[g] = d
                del locus_to_genes[m]
            locusname = '+'.join(sorted(toks))
            locus_gene_distance = [(g, d) for (g, d) in all_genes_dist.items()]
            locus_to_genes[locusname] = sorted(
                locus_gene_distance, key=lambda x: x[1])

            for g, d in all_genes_dist.items():
                gene_to_locus[g] = (locusname, d)

    return locus_to_genes


def merge_loci(union_locus_geneset, union_locus_gene_distance):

    locus_to_genes = defaultdict(list)  # genes as a list
    gene_to_locus = dict()  # gene points to a single locus
    for locus in union_locus_geneset:
        genes = union_locus_geneset[locus]
        # figure out if any of the genes is already in a locus
        merge = dict()
        for g in genes:
            if g in gene_to_locus:
                merge[gene_to_locus[g][0]] = True

        if not merge:
            for g in genes:
                d = abs(union_locus_gene_distance[locus][g])
                gene_to_locus[g] = (locus, d)
                locus_to_genes[locus].append((g, d))
        else:
            all_genes_dist = dict()
            for g in genes:

                d = abs(union_locus_gene_distance[locus][g])
                all_genes_dist[g] = d

            toks = [locus]
            for m in merge:
                toks = toks + m.split('+')
                other_snp_genes = [x[0] for x in locus_to_genes[m]]
                for g in other_snp_genes:
                    d = gene_to_locus[g][1]
                    if g in all_genes_dist:
                        d = min(all_genes_dist[g], d)
                    all_genes_dist[g] = d
                del locus_to_genes[m]
            snpset = set()
            for t in toks:
                for s in t.split('+'):
                    snpset.add(s)
            # locusname = '+'.join(sorted(toks))
            locusname = '+'.join(sorted(snpset))
            locus_gene_distance = [(g, d) for (g, d) in all_genes_dist.items()]
            locus_to_genes[locusname] = sorted(
                locus_gene_distance, key=lambda x: x[1])

            for g, d in all_genes_dist.items():
                gene_to_locus[g] = (locusname, d)

    return locus_to_genes


def GWAScat_rows(phenotype_list, GWAS_dir, TopMedData_dir):
    for phenotype in phenotype_list:
        filename = phenotype + '_GWAS_rows.tsv'
        inputfile = os.path.join(GWAS_dir, filename)

        df = pd.read_csv(inputfile, delimiter='\t')

        rsid_df = df[['SNPS']]
        rsid_list = df['SNPS'].values

        # extract variant rsid in a file to be uploaded on Ensembl
        outfile = GWAS_dir + phenotype + '_GWAScat_rsid.txt'
        with open(outfile, 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(rsid_list)

        # extract SNP data
        GWAScat_df = df[['SNPS', 'PUBMEDID', 'MAPPED_TRAIT', 'STRONGEST SNP-RISK ALLELE',
                         'P-VALUE', 'OR or BETA', '95% CI (TEXT)', 'CONTEXT']]

        GWAScat_df['Risk_Allele'] = [
            str.split(x, '-')[1] for x in GWAScat_df['STRONGEST SNP-RISK ALLELE'].values]
        logger.info('writing %s', filename)
        GWAScat_df.to_csv(path_or_buf=TopMedData_dir +
                          phenotype + '_GWAScat.txt', sep='\t')
    return None

"""
This script benchmarks EVA-generated miRNAs by
1. read in ~/projects/EVA/output/eva_human_mirna.csv and Manakov2022/benchmark_mirarchitect.csv
2. for each gene in benchmar_mirarchitect.csv, randomly sample 20 sequences from eva_human_mirna.csv
3. score the sampled miRNA using RNAhybrid and pick the best one as the EVA-generated miRNA for that gene
4. score the EVA-generated mirna-gene pairs using Discriminator, RNACofold, and RNAHybrid
5. save the scores to a csv file ~/projects/EVA/output/eva_benchmark_scores.csv
"""

import pandas as pd
import numpy as np
import os

from Global_parameters import PROJ_HOME, RNAHYBRID_BIN, RNACOFOLD_BIN
from Table2Three_scorers import Discriminator_scorer, RNACofold_scorer, RNAHybrid_scorer
from thermodynamics_utils import RNAHybrid_score

if __name__ == "__main__":
    eva_mirna_path = os.path.expanduser("~/projects/EVA/output/eva_human_mirna.csv")
    eva_mirna_df = pd.read_csv(eva_mirna_path)
    benchmark_mirarchitect_path = os.path.join(PROJ_HOME, "Manakov2022/benchmark_mirarchitect.csv")
    benchmark_mirarchitect_df = pd.read_csv(benchmark_mirarchitect_path)
    discriminator_scorer = Discriminator_scorer()
    rnacofold_scorer = RNACofold_scorer()
    rnahybrid_scorer = RNAHybrid_scorer()
    eva_df = {"gene": [], "generated_mirna": [], "noncodingRNA": []}
    for idx, row in benchmark_mirarchitect_df.iterrows():
        gene = row["gene"]
        noncodingRNA = row["noncodingRNA"]
        sampled_mirnas = eva_mirna_df['sequence'].sample(n=50).to_list()
        RNAHybrid_scores = [RNAHybrid_score(mirna_seq, gene) for mirna_seq in sampled_mirnas]
        best_mirna_seq = sampled_mirnas[np.argmax(RNAHybrid_scores)]
        print(f"Best EVA miRNA for {gene}: {best_mirna_seq}; Best RNAHybrid score: {np.max(RNAHybrid_scores)}")
        eva_df["gene"].append(gene)
        eva_df["generated_mirna"].append(best_mirna_seq)
        eva_df["noncodingRNA"].append(noncodingRNA)
    eva_df = pd.DataFrame(eva_df)
    eva_df_path = os.path.join(os.path.dirname(eva_mirna_path), "eva_best_benchmark.csv")
    eva_df.to_csv(eva_df_path, index=False)
    print(f"EVA scores saved to {eva_df_path}")

    # read in the off_target dataset and the mirarchitect dataset
    off_target_path = os.path.join(PROJ_HOME, "Manakov2022/benchmark_mirarchitect_off_targets.tsv.gz")
    off_target_df = pd.read_csv(off_target_path, sep='\t', compression='gzip')
    eva_df = pd.read_csv(eva_df_path)
    # replace the generated_mirna in the off_target dataset with the generated_mirna in the mirarchitect dataset
    for i, row in off_target_df.iterrows():
        gene = row["gene"]
        off_target_gene = row["off_target_gene"]
        noncodingRNA = row["noncodingRNA"]
        generated_mirna = eva_df.loc[eva_df['gene'] == gene, "generated_mirna"].values[0]
        off_target_df.loc[i, "generated_mirna"] = generated_mirna
    eva_off_target_path = os.path.join(os.path.dirname(eva_df_path), "eva_off_targets.tsv.gz")
    off_target_df.to_csv(eva_off_target_path, sep='\t', compression='gzip', index=False)
    print(f"EVA off-targets saved to {eva_off_target_path}")

"""
This script finds off-targets for every noncodingRNA in Manakov2022/benchmark_mirarchitect.csv.

1. Take the reverse complement of the 2–8 nt seed region of the noncodingRNA.
2. Walk the pool in file order; collect up to MAX_SEED_MATCH_CANDIDATES passing hits, then sample K_OFF at random from that set (excludes pairing gene and cluster).
3. Save gene, off-target gene, noncodingRNA, and generated_miRNA to a gzipped TSV.
"""

import os
import random
import pandas as pd
from Global_parameters import PROJ_HOME

K_OFF = 20
MAX_SEED_MATCH_CANDIDATES = 1000
OFF_TARGET_RNG_SEED = 42

def reverse_complement(seq):
    # reverse the sequence and complement the base
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(comp.get(b) for b in reversed(seq))

def find_seed_matched_off_targets(
    target_seed_region,
    test_mrnas_df,
    target_genes=None,
    target_clusters=None,
    max_samples=K_OFF,
    max_candidates=MAX_SEED_MATCH_CANDIDATES,
    rng=None,
):
    """
    Collect up to max_candidates matching genes, then sample up to max_samples
    of them without replacement.
    """
    if target_genes is None:
        target_genes = ()
    if target_clusters is None:
        target_clusters = ()

    candidates = []
    for _, row in test_mrnas_df.iterrows():
        mrna_seq = row["gene"]
        if (
            row["gene_cluster_ID"] not in target_clusters
            and mrna_seq not in target_genes
            and target_seed_region in mrna_seq
        ):
            candidates.append(mrna_seq)
            if len(candidates) >= max_candidates:
                break

    if not candidates:
        return []
    k = min(max_samples, len(candidates))
    if rng is None:
        rng = random.Random()
    return rng.sample(candidates, k=k)

if __name__ == "__main__":
    # ── Paths ──────────────────────────────────────────────────────────────────────
    benchmark_mirarchitect_path = os.path.join(PROJ_HOME, "Manakov2022/benchmark_mirarchitect.csv")
    off_target_pool_path = os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test.tsv.gz")
    mirarchitect_df_path = os.path.join(PROJ_HOME, "mirarchitect_data/mirarchitect-amirnas.csv")
    save_path = os.path.join(PROJ_HOME, "Manakov2022/benchmark_mirarchitect_off_targets.tsv.gz")

    # ── Load data ───────────────────────────────────────────────────────────────────
    benchmark_mirarchitect_df = pd.read_csv(benchmark_mirarchitect_path)
    off_target_pool = pd.read_csv(off_target_pool_path, sep='\t', compression='gzip')

    # ── Find off-targets ────────────────────────────────────────────────────────────
    off_target_dict = {"gene": [], "off_target_gene": [], "noncodingRNA": [], "generated_mirna": []}
    rng = random.Random(OFF_TARGET_RNG_SEED)
    for i, row in benchmark_mirarchitect_df.iterrows():
        noncodingRNA = row['noncodingRNA']
        generated_mirna = row['generated_mirna']
        seed_region = noncodingRNA[1:8]
        seed_region_rc = reverse_complement(seed_region)
        target_gene = row["gene"]
        gene_cluster_ID = row["gene_cluster_ID"]
        off_targets = find_seed_matched_off_targets(
            target_seed_region=seed_region_rc,
            test_mrnas_df=off_target_pool,
            target_genes=[target_gene],
            target_clusters=[gene_cluster_ID],
            max_samples=K_OFF,
            rng=rng,
        )
        print(f"Found {len(off_targets)} off-targets for {noncodingRNA}")
        for off_target in off_targets:
            off_target_dict["gene"].append(target_gene)
            off_target_dict["off_target_gene"].append(off_target)
            off_target_dict["noncodingRNA"].append(noncodingRNA)
            off_target_dict["generated_mirna"].append(generated_mirna)
    off_target_df = pd.DataFrame(off_target_dict)
    off_target_df.to_csv(save_path, sep="\t", compression="gzip", index=False)
    print(f"Off-targets saved to {save_path}")

    # read in the off_target dataset and the mirarchitect dataset
    off_target_df = pd.read_csv(save_path, sep='\t', compression='gzip')
    mirarchitect_df = pd.read_csv(mirarchitect_df_path)
    # replace the generated_mirna in the off_target dataset with the generated_mirna in the mirarchitect dataset
    for i, row in off_target_df.iterrows():
        gene = row["gene"]
        off_target_gene = row["off_target_gene"]
        noncodingRNA = row["noncodingRNA"]
        generated_mirna = mirarchitect_df.loc[mirarchitect_df['gene'] == gene, "generated_mirna"].values[0]
        off_target_df.loc[i, "generated_mirna"] = generated_mirna
    save_path = os.path.join(PROJ_HOME, "mirarchitect_data/mirarchitect_off_targets.tsv.gz")
    off_target_df.to_csv(save_path, sep='\t', compression='gzip', index=False)
    print(f"Off-targets saved to {save_path}")

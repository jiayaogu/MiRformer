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
    comp = {"A": "T", "T": "A", "U": "A", "C": "G", "G": "C"}
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

    target_genes = set(target_genes)
    target_clusters = set(target_clusters)
    mask = (
        ~test_mrnas_df["gene_cluster_ID"].isin(target_clusters)
        & ~test_mrnas_df["gene"].isin(target_genes)
        & test_mrnas_df["gene"].str.contains(target_seed_region, regex=False, na=False)
    )
    candidates = test_mrnas_df.loc[mask, "gene"].head(max_candidates).tolist()

    if not candidates:
        return []
    k = min(max_samples, len(candidates))
    if rng is None:
        rng = random.Random()
    return rng.sample(candidates, k=k)

if __name__ == "__main__":
    # ── Paths ──────────────────────────────────────────────────────────────────────
    benchmark_mirarchitect_path = os.path.join(PROJ_HOME, "mirarchitect_data/specimir_mirna.csv")
    off_target_pool_path = os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test.tsv.gz")
    mirarchitect_df_path = os.path.join(PROJ_HOME, "mirarchitect_data/mirarchitect-amirnas.csv")
    save_path = os.path.join(PROJ_HOME, "mirarchitect_data/specimir_mirna_off_targets.csv")

    # ── Load data ───────────────────────────────────────────────────────────────────
    benchmark_mirarchitect_df = pd.read_csv(benchmark_mirarchitect_path, dtype=str)
    mirarchitect_df = pd.read_csv(mirarchitect_df_path, sep=',', dtype=str)
    off_target_pool = pd.read_csv(
        off_target_pool_path,
        sep='\t',
        compression='gzip',
        dtype=str,
    )

    # ── Find off-targets ────────────────────────────────────────────────────────────
    specimir_off_target_rows = []
    mirarchitect_off_target_rows = []
    mirarchitect_by_gene = {
        row["gene"]: row.to_dict()
        for _, row in mirarchitect_df.iterrows()
    }
    rng = random.Random(OFF_TARGET_RNG_SEED)
    for i, row in benchmark_mirarchitect_df.iterrows():
        noncodingRNA = row['noncodingRNA']
        seed_region = noncodingRNA[1:8]
        seed_region_rc = reverse_complement(seed_region)
        target_gene = row["gene"]
        gene_cluster_ID = row["gene_cluster_ID"]
        mirarchitect_row = mirarchitect_by_gene.get(target_gene)
        if mirarchitect_row is None:
            raise ValueError(f"No mirarchitect row found for gene: {target_gene}")
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
            specimir_row = row.to_dict()
            specimir_row["off_target_gene"] = off_target
            specimir_off_target_rows.append(specimir_row)

            mirarchitect_off_target_row = dict(mirarchitect_row)
            mirarchitect_off_target_row["off_target_gene"] = off_target
            mirarchitect_off_target_rows.append(mirarchitect_off_target_row)

    off_target_df = pd.DataFrame(specimir_off_target_rows)
    off_target_df.to_csv(save_path, sep=",", index=False)
    print(f"Off-targets saved to {save_path}")

    save_path = os.path.join(PROJ_HOME, "mirarchitect_data/mirarchitect_off_targets.csv")
    off_target_df = pd.DataFrame(mirarchitect_off_target_rows)
    off_target_df.to_csv(save_path, sep=',', index=False)
    print(f"Off-targets saved to {save_path}")

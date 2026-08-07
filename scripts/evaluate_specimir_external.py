#!/usr/bin/env python
"""External evaluation of SpeciMiR-generated miRNAs using the miRBench predictor panel.

Scores four separate (miRNA, mRNA) datasets with each predictor:
    1. real miRNA      <-> on-target mRNA
    2. generated miRNA <-> on-target mRNA
    3. generated miRNA <-> off-target mRNA
    4. random miRNA    <-> on-target mRNA

None of these predictors is the SpeciMiR discriminator, so the resulting
on/off separation is an external measurement of specificity.

Environment (miRBench CPU stack; do not mix with HyenaDNA_clone2):
    conda activate /path/to/MiRformer/mirbench_venv
    # or: source mirbench_venv/bin/activate
Docs: https://github.com/katarinagresova/miRBench
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from miRBench.predictor import get_predictor
from miRBench.encoder import get_encoder

from Global_parameters import PROJ_HOME

# ---------------------------------------------------------------- config
# Tier 1 = different assay AND readout from an AGO-eCLIP binding classifier.
# Tier 2 = independently trained binding model, different dataset.
# Tier 3 = trained on Manakov/Hejret AGO2 eCLIP -> NOT independent of D_phi.
TIER1 = ["TargetScanCnn_McGeary2019", "Seed8mer", "Seed7mer",
         "Seed6mer", "Seed6merBulgeOrMismatch"]
TIER2 = ["TargetNet_Min2021", "InteractionAwareModel_Yang2024",
         "CnnMirTarget_Zheng2020", "miRBind_Klimentova2022"]
TIER3 = ["miRNA_CNN_Hejret2023", "miRBenchCNN_HejretCorrected"]

PANEL = TIER1 + TIER2
TIER_OF = {**{p: 1 for p in TIER1}, **{p: 2 for p in TIER2}, **{p: 3 for p in TIER3}}

# Higher score = stronger predicted interaction, EXCEPT TargetScanCnn which
# returns relative K_D (dissociation constant): LOWER = tighter binding.
LOWER_IS_BETTER = {"TargetScanCnn_McGeary2019"}

SEED = 42
N_SUBSAMPLE = 2000  # match Table2Three_scorers / off-target & random set sizes
OUT_DIR = os.path.join(PROJ_HOME, "Manakov2022")

# On-target pairs come from the SpeciMiR prediction TSV; off-target / random
# sets are precomputed (same subsample seed as Table2Three_scorers).
PREDICTED_TSV = os.path.join(OUT_DIR, "AGO2_eCLIP_Manakov2022_test_predicted.tsv")
GEN_OFF_TSV = os.path.join(OUT_DIR, "AGO2_eCLIP_Manakov2022_test_predicted_generated_off_target.tsv")
RANDOM_TSV = os.path.join(OUT_DIR, "AGO2_eCLIP_Manakov2022_test_predicted_random.tsv")


def load_datasets():
    """Load the four (miRNA, mRNA) evaluation sets as separate DataFrames.

    Each returned frame has columns ``gene`` (mRNA) and ``mirna``.
    """
    predicted = pd.read_csv(PREDICTED_TSV, sep="\t")
    predicted = predicted.sample(n=N_SUBSAMPLE, random_state=SEED, ignore_index=True)
    predicted = predicted[
        (predicted["label"] == 1)
        & (predicted["generated_mirna"] != "NA")
        & (predicted["noncodingRNA"] != "NA")
    ]

    real_mirna_to_mrna = (
        predicted[["gene", "noncodingRNA"]]
        .rename(columns={"noncodingRNA": "mirna"})
        .reset_index(drop=True)
    )
    gen_mirna_to_mrna = (
        predicted[["gene", "generated_mirna"]]
        .rename(columns={"generated_mirna": "mirna"})
        .reset_index(drop=True)
    )
    gen_off_dataset = (
        pd.read_csv(GEN_OFF_TSV, sep="\t")[["gene", "mirna"]]
        .reset_index(drop=True)
    )
    random_dataset = (
        pd.read_csv(RANDOM_TSV, sep="\t")[["gene", "mirna"]]
        .reset_index(drop=True)
    )

    return {
        "real_miRNA_target_mRNA": real_mirna_to_mrna,
        "generated_miRNA_target_mRNA": gen_mirna_to_mrna,
        "generated_miRNA_off_target_mRNA": gen_off_dataset,
        "random_miRNA_target_mRNA": random_dataset,
    }


def score_panel(df, panel=PANEL, mirna_col="mirna", gene_col="gene"):
    """df needs columns [mirna_col, gene_col]; returns df with one column per predictor."""
    out = df.copy()
    for name in panel:
        print(f"  scoring {name} (n={len(df)}) ...", flush=True)
        enc = get_encoder(name)
        pred = get_predictor(name)
        x = enc(df, miRNA_col=mirna_col, gene_col=gene_col)
        out[name] = np.asarray(pred(x)).ravel()
    return out


def summary_table(scored_by_dataset):
    """Mean / std of each predictor on each dataset."""
    rows = []
    for dataset_name, scored in scored_by_dataset.items():
        for name in PANEL:
            vals = scored[name].to_numpy()
            if name in LOWER_IS_BETTER:
                vals = -vals
            rows.append(dict(
                dataset=dataset_name,
                predictor=name,
                tier=TIER_OF[name],
                n=len(vals),
                mean=vals.mean(),
                std=vals.std(),
            ))
    return pd.DataFrame(rows).sort_values(["dataset", "tier", "predictor"])



if __name__ == "__main__":
    datasets = load_datasets()
    for name, df in datasets.items():
        print(f"{name}: {len(df)} pairs")

    scored_by_dataset = {}
    for dataset_name, dataset in datasets.items():
        print(f"\n=== {dataset_name} ===", flush=True)
        scored = score_panel(dataset)
        scored_by_dataset[dataset_name] = scored
        out_path = os.path.join(OUT_DIR, f"specimir_external_scores_{dataset_name}.tsv")
        scored.to_csv(out_path, sep="\t", index=False)
        print(f"Saved {out_path}")

    summary = summary_table(scored_by_dataset)
    summary_path = os.path.join(OUT_DIR, "specimir_external_summary.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSummary:\n{summary.to_string(index=False)}")
    print(f"Saved {summary_path}")

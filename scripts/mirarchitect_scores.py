"""
This script scores the mirarchitect-designed miRNAs using three scorers:
Discriminator, RNACofold, and RNAHybrid.
"""

import os
import pandas as pd
import numpy as np

from Global_parameters import PROJ_HOME
from Table2Three_scorers import Discriminator_scorer, RNACofold_scorer, RNAHybrid_scorer

if __name__ == "__main__":
    data_path = os.path.join(PROJ_HOME, "mirarchitect_data/mirarchitect_off_targets.tsv.gz")
    df = pd.read_csv(data_path, sep='\t', compression='gzip')
    discriminator_scorer = Discriminator_scorer()
    rnacofold_scorer = RNACofold_scorer()
    rnahybrid_scorer = RNAHybrid_scorer()
    # score targets
    discriminator_scores = discriminator_scorer.score_batch(df, mirna_col="generated_mirna", mrna_col="gene")
    rnacofold_scores = rnacofold_scorer.score_batch(df, mirna_col="generated_mirna", mrna_col="gene")
    rnahybrid_scores = rnahybrid_scorer.score_batch(df, mirna_col="generated_mirna", mrna_col="gene")
    # score off-targets
    off_target_discriminator_scores = discriminator_scorer.score_batch(df, mirna_col="generated_mirna", mrna_col="off_target_gene")
    off_target_rnacofold_scores = rnacofold_scorer.score_batch(df, mirna_col="generated_mirna", mrna_col="off_target_gene")
    off_target_rnahybrid_scores = rnahybrid_scorer.score_batch(df, mirna_col="generated_mirna", mrna_col="off_target_gene")
    # summary statistics (target)
    mean_discriminator_score = np.mean(discriminator_scores)
    std_discriminator_score = np.std(discriminator_scores)
    mean_rnacofold_score = np.mean(rnacofold_scores)
    std_rnacofold_score = np.std(rnacofold_scores)
    mean_rnahybrid_score = np.mean(rnahybrid_scores)
    std_rnahybrid_score = np.std(rnahybrid_scores)
    # summary statistics (off-target)
    mean_off_target_discriminator_score = np.mean(off_target_discriminator_scores)
    std_off_target_discriminator_score = np.std(off_target_discriminator_scores)
    mean_off_target_rnacofold_score = np.mean(off_target_rnacofold_scores)
    std_off_target_rnacofold_score = np.std(off_target_rnacofold_scores)
    mean_off_target_rnahybrid_score = np.mean(off_target_rnahybrid_scores)
    std_off_target_rnahybrid_score = np.std(off_target_rnahybrid_scores)
    print(f"Mean discriminator score: {mean_discriminator_score}, std: {std_discriminator_score}")
    print(f"Mean rnacofold score: {mean_rnacofold_score}, std: {std_rnacofold_score}")
    print(f"Mean rnahybrid score: {mean_rnahybrid_score}, std: {std_rnahybrid_score}")
    print(f"Mean off-target discriminator score: {mean_off_target_discriminator_score}, std: {std_off_target_discriminator_score}")
    print(f"Mean off-target rnacofold score: {mean_off_target_rnacofold_score}, std: {std_off_target_rnacofold_score}")
    print(f"Mean off-target rnahybrid score: {mean_off_target_rnahybrid_score}, std: {std_off_target_rnahybrid_score}")
    
    # save the scores to a csv file
    scores_df = pd.DataFrame({
        "gene": df["gene"],
        "noncodingRNA": df["noncodingRNA"],
        "discriminator_score": discriminator_scores,
        "rnacofold_score": rnacofold_scores,
        "rnahybrid_score": rnahybrid_scores,
        "discriminator_off_target_score": off_target_discriminator_scores,
        "rnacofold_off_target_score": off_target_rnacofold_scores,
        "rnahybrid_off_target_score": off_target_rnahybrid_scores
    })
    save_path = os.path.join(PROJ_HOME, "mirarchitect_data/mirarchitect_off_target_scores.tsv.gz")
    scores_df.to_csv(save_path, sep='\t', compression='gzip', index=False)
    print(f"Scores saved to {save_path}")
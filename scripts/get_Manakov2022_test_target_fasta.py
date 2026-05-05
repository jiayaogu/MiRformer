import os
import pandas as pd
from Global_parameters import PROJ_HOME

data_path = os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test_predicted.tsv.gz")
out_path = os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test_predicted.fasta")

df = pd.read_csv(data_path, sep="\t", usecols=["gene", "gene_cluster_ID"])
df = df.sample(n=2000, random_state=42)

with open(out_path, "w") as f:
    for _, row in df.iterrows():
        f.write(f">{row['gene_cluster_ID']}\n{row['gene']}\n")

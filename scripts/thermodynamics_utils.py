import subprocess
import numpy as np
from Global_parameters import RNACOFOLD_BIN, RNAHYBRID_BIN


def RNACofold_score(mirna_seq, mrna_seq, rnacofold_bin=RNACOFOLD_BIN):
    """
    Score the duplex stability of the miRNA-mRNA using RNACofold
    """
    mirna_seq = mirna_seq.replace("T", "U")
    mrna_seq = mrna_seq.replace("T", "U")
    input_line = "&".join([mirna_seq, mrna_seq])

    try:
        results = subprocess.run(
            # -a partition function including delta G binding, -d2 dangling end energy treatment, 
            # --noLP no long-range pairing
            [rnacofold_bin, "-a", "-d2"], 
            input=input_line,
            capture_output=True,
            text=True,
            timeout=30
        )
        if results.returncode != 0:
            print(f"RNACofold failed: {results.stderr}", flush=True)
            return np.nan

        output_lines = [l for l in results.stdout.splitlines() if l.strip()]
        for line in output_lines:
            if "delta G binding=" in line:
                # "... delta G binding= -5..."
                delta_g_binding = float(line.split("delta G binding=")[1].strip().split(" ")[0])
                return delta_g_binding
        
        print(f"RNACofold output missing delta G binding: {results.stdout}", flush=True)
        return np.nan

    except Exception as e:
        print(f"Error running RNACofold: {e}", flush=True)
        return np.nan
    
def RNAHybrid_score(mirna_seq, mrna_seq, rnahybrid_bin=RNAHYBRID_BIN):
    """
    Score the change in free energy of the miRNA-mRNA duplex using RNAhybrid
    """
    mirna_seq = mirna_seq.replace("T", "U")
    mrna_seq = mrna_seq.replace("T", "U")

    try:
        results = subprocess.run(
            [rnahybrid_bin, 
            "-c",       # compact output
            "-b", "1",  # 1 hit per target
            "-d", "1.28, 0.9", # manually provide xi and theta parameters for calculating P-value of MFE, will not affect the MFE score
            mrna_seq,   # target (long mRNA)
            mirna_seq,  # query  (short miRNA)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        mfe_str = results.stdout.split(":")[4] # "command_line:100:command_line:22:-21.4..."
        mfe = float(mfe_str)
        return mfe
    except Exception as e:
        print(f"Error running RNAhybrid: {e}", flush=True)
        return np.nan
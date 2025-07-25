from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import argparse



# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/duke_lcs/step2_create_split.py --csv_path /radraid2/mwahianwar/duke_lcs/data/DLCSD24_Annotations_v0.csv 
# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/duke_lcs/step2_create_split.py

# patient-id,nodule_id,ct_nifti_file,coordX,coordY,coordZ,w,h,d,Malignant_lbl,benchmark_split
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/radraid2/mwahianwar/duke_lcs/data/DLCSD24_Annotations.csv', type=str)
    parser.add_argument('--output_dir', default='/radraid2/mwahianwar/MST/duke_lcs', type=str)

    args = parser.parse_args()

    # === Paths ===
    csv_path = Path(args.csv_path)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # === Load and prepare DataFrame ===
    df = pd.read_csv(csv_path)
    print("Loaded:", len(df), "rows")

    # Drop incomplete rows if necessary
    df = df.dropna(subset=["patient-id", "nodule_id", "Malignant_lbl"])

    # Treat `label` as the malignancy label
    df['Malignant'] = df['Malignant_lbl'].astype(int)

    # Check class balance
    print("Malignant counts:\n", df['Malignant_lbl'].value_counts())

    # Ensure one row per unique lesion
    unique_cols = ["patient-id", "nodule_id"]
    df_unique = df.drop_duplicates(subset=unique_cols).copy()

    # Prepare stratified split
    df_unique = df_unique.reset_index(drop=True)
    splits = []
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_i, (trainval_idx, test_idx) in enumerate(
        sgkf.split(df_unique, df_unique['Malignant_lbl'], groups=df_unique['patient-id'])
    ):
        df_split = df_unique.copy()
        df_split['Fold'] = fold_i

        df_trainval = df_split.loc[trainval_idx]
        train_idx, val_idx = next(sgkf2.split(df_trainval, df_trainval['Malignant_lbl'], groups=df_trainval['patient-id']))
        train_idx, val_idx = df_trainval.iloc[train_idx].index, df_trainval.iloc[val_idx].index

        df_split['Split'] = 'ignore'
        df_split.loc[train_idx, 'Split'] = 'train'
        df_split.loc[val_idx, 'Split'] = 'val'
        df_split.loc[test_idx, 'Split'] = 'test'

        splits.append(df_split)

    df_splits = pd.concat(splits, ignore_index=True)


    # Drop original Malignant column to avoid _x/_y suffix conflict
    df = df.drop(columns=["Malignant"], errors="ignore")

    df_out = pd.merge(df, df_splits[[*unique_cols, "Malignant", "Fold", "Split"]],
                    on=unique_cols, how="left")
    # Save
    output_csv = output_path / "split.csv"
    df_out.to_csv(output_csv, index=False)
    print(f"Saved split CSV to {output_csv}")

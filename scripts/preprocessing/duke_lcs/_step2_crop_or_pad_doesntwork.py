from pathlib import Path
import torchio as tio
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

print("TorchIO version:", tio.__version__)
print("CropOrPad init signature:", tio.CropOrPad.__init__.__annotations__)

def crop_case(uid_df_tuple):
    uid, rows = uid_df_tuple
    img_path = image_dir / f"{uid}.nii.gz"
    if not img_path.exists():
        return f"Missing image: {img_path}"

    try:
        image = tio.ScalarImage(img_path)
    except Exception as e:
        return f"Failed to load image {uid}: {e}"

    affine = image.affine
    inv_affine = np.linalg.inv(affine)

    errors = []

    for idx, row in rows.iterrows():
        x, y, z = row["CoordX"], row["CoordY"], row["CoordZ"]
        lesion_id = row.get("LesionID", idx)
        physical_coords = np.array([[x], [y], [z], [1.0]])
        voxel_coords = inv_affine @ physical_coords
        voxel_center = voxel_coords[:3].flatten()

        subject = tio.Subject(img=image)
        transform = tio.CropOrPad(
            target_shape=crop_shape,
            padding_mode=padding_value,
            crop_center=tuple(voxel_center.tolist())
        )

        try:
            cropped = transform(subject)
            out_dir = output_dir / uid
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"img_{lesion_id}.nii.gz"
            cropped["img"].save(out_path)
        except Exception as e:
            errors.append(f"Failed to crop {uid} at physical {x,y,z} (voxel {voxel_center.tolist()}): {e}")

    return errors if errors else None


# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/luna25/step2_crop_or_pad.py --sequential 
# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/luna25/step2_crop_or_pad.py --num_workers 20 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/radraid2/mwahianwar/miccai25/luna25challenge/data/LUNA25_Public_Training_Development_Data.csv', type=str)
    parser.add_argument('--preprocessed_image_dir', default='/radraid2/mwahianwar/MST/luna25/preprocessed', type=str)
    parser.add_argument('--output_dir', default='/radraid2/mwahianwar/MST/luna25/preprocessed_crop', type=str)
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel workers")
    parser.add_argument('--sequential', action='store_true', help="Run sequentially instead of parallel")

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    image_dir = Path(args.preprocessed_image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["SeriesInstanceUID", "CoordX", "CoordY", "CoordZ"])
    grouped = list(df.groupby("SeriesInstanceUID"))  # Convert to list of (uid, DataFrame)

    crop_shape = (256, 256, 32)
    padding_value = -1024

    errors = []

    if args.sequential:
        for uid_df in tqdm(grouped, desc="Cropping (Sequential)"):
            err = crop_case(uid_df)
            if err:
                errors.extend(err if isinstance(err, list) else [err])
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(crop_case, uid_df): uid_df[0] for uid_df in grouped}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Cropping (Parallel)"):
                result = future.result()
                if result:
                    errors.extend(result if isinstance(result, list) else [result])

    if errors:
        print("\nSome crops failed:")
        for e in errors:
            print(e)
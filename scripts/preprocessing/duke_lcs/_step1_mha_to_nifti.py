import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse


def convert_case(uid, image_dir, output_dir):
    input_path = image_dir / f"{uid}.mha"
    output_path = output_dir / f"{uid}.nii.gz"

    if not input_path.exists():
        return f"Warning: {input_path} does not exist."

    try:
        image = sitk.ReadImage(str(input_path))
        sitk.WriteImage(image, str(output_path))
        return None
    except Exception as e:
        return f"Failed to convert {uid}: {e}"


### python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/luna25/step1_mha_to_nifti.py --num_workers 20 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/radraid2/mwahianwar/duke_lcs/data/DLCSD24_CT_ImageInfo_v1.csv', type=str)
    parser.add_argument('--image_dir', default='/radraid2/mwahianwar/miccai25/luna25challenge/data/luna25_images', type=str)
    parser.add_argument('--output_dir', default='/radraid2/mwahianwar/MST/luna25/preprocessed', type=str)
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel processes (only used if not sequential)")
    parser.add_argument('--sequential', action='store_true', help="Run conversion sequentially instead of in parallel")

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    unique_uids = df["SeriesInstanceUID"].dropna().unique()

    errors = []

    if args.sequential:
        for uid in tqdm(unique_uids, desc="Sequentially converting MHA to NIfTI"):
            result = convert_case(uid, image_dir, output_dir)
            if result:
                errors.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(convert_case, uid, image_dir, output_dir): uid for uid in unique_uids
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting MHA to NIfTI (Parallel)"):
                result = future.result()
                if result:
                    errors.append(result)

    if errors:
        print("\nSome conversions failed:")
        for e in errors:
            print(e)
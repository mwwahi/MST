import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def convert_case(uid, image_dir, output_dir):
    input_path = image_dir / f"{uid}.nii.gz"
    
    if not input_path.exists():
        return f"Warning: {input_path} does not exist."
    
    try:
        image = sitk.ReadImage(str(input_path))
        
        # FIXED: Create output directory structure: output_dir/data/SeriesInstanceUID/
        # Note: output_dir should be the base directory, not include /data/ already
        series_output_dir = output_dir / uid
        series_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 1: Single img.nii.gz file per SeriesInstanceUID
        output_path = series_output_dir / "img.nii.gz"
        sitk.WriteImage(image, str(output_path))
        return None
        
    except Exception as e:
        return f"Failed to convert {uid}: {e}"

### Usage:
# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/duke_lcs/step0_move_nifti.py --debug
# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/duke_lcs/step0_move_nifti.py --sequential
# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/duke_lcs/step0_move_nifti.py --num_workers 20 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" Moving NIfTI file")
    parser.add_argument('--csv_path', default='/radraid2/mwahianwar/duke_lcs/data/DLCSD24_Annotations.csv', type=str)
    parser.add_argument('--image_dir', default='/radraid2/mwahianwar/duke_lcs/data/combined_data', type=str)
    # FIXED: Changed default output directory to avoid permission issues
    parser.add_argument('--output_dir', default='/radraid2/mwahianwar/MST/duke_lcs/preprocessed', type=str)
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel processes")
    parser.add_argument('--sequential', action='store_true', help="Run conversion sequentially")
    parser.add_argument('--debug', action='store_true', help="Debug mode - process only first series")
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    
    # Check if we can write to the output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"ERROR: No permission to create directory: {output_dir}")
        print(f"Try running with: --output_dir ./preprocessed")
        exit(1)
    
    # Read CSV and get unique SeriesInstanceUIDs
    try:
        df = pd.read_csv(csv_path)
        unique_uids = df["patient-id"].dropna().unique()
    except Exception as e:
        print(f"ERROR: Could not read CSV file: {csv_path}")
        print(f"Error: {e}")
        exit(1)
    
    print(f"Phase 1 Conversion Summary:")
    print(f"Total series: {len(unique_uids)}")
    print(f"Input directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output structure: {output_dir}/data/[SeriesInstanceUID]/img.nii.gz")
    
    # Check if input directory exists
    if not image_dir.exists():
        print(f"ERROR: Input directory does not exist: {image_dir}")
        exit(1)
    
    # Count available .mha files
    nifti_files = list(image_dir.glob("*.nii.gz"))
    print(f"Found {len(nifti_files)} .nii.gz files in input directory")

    errors = []
    
    if args.sequential:
        for uid in tqdm(unique_uids, desc="Sequentially converting MHA to NIfTI (Phase 1)"):
            result = convert_case(uid, image_dir, output_dir)
            if result:
                errors.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(convert_case, uid, image_dir, output_dir): uid 
                for uid in unique_uids
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Moving nifti images"):
                result = future.result()
                if result:
                    errors.append(result)

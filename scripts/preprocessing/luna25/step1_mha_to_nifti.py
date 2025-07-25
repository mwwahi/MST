import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def convert_case(uid, image_dir, output_dir):
    input_path = image_dir / f"{uid}.mha"
    
    if not input_path.exists():
        return f"Warning: {input_path} does not exist."
    
    try:
        image = sitk.ReadImage(str(input_path))
        
        # FIXED: Create output directory structure: output_dir/data/SeriesInstanceUID/
        # Note: output_dir should be the base directory, not include /data/ already
        series_output_dir = output_dir / "data" / uid
        series_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 1: Single img.nii.gz file per SeriesInstanceUID
        output_path = series_output_dir / "img.nii.gz"
        sitk.WriteImage(image, str(output_path))
        return None
        
    except Exception as e:
        return f"Failed to convert {uid}: {e}"

### Usage:
# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/luna25/step1_mha_to_nifti.py --debug
# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/luna25/step1_mha_to_nifti.py --sequential
# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/luna25/step1_mha_to_nifti.py --num_workers 20 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Convert MHA to single NIfTI file per series")
    parser.add_argument('--csv_path', default='/radraid2/mwahianwar/miccai25/luna25challenge/data/LUNA25_Public_Training_Development_Data.csv', type=str)
    parser.add_argument('--image_dir', default='/radraid2/mwahianwar/miccai25/luna25challenge/data/luna25_images', type=str)
    # FIXED: Changed default output directory to avoid permission issues
    parser.add_argument('--output_dir', default='/radraid2/mwahianwar/MST/luna25/preprocessed', type=str)
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
        unique_uids = df["SeriesInstanceUID"].dropna().unique()
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
    mha_files = list(image_dir.glob("*.mha"))
    print(f"Found {len(mha_files)} .mha files in input directory")
    
    # Debug mode - process only first series
    if args.debug:
        if len(unique_uids) > 0:
            first_uid = unique_uids[0]
            print(f"\nDEBUG: Processing only first series: {first_uid}")
            print(f"DEBUG: Input path: {image_dir / f'{first_uid}.mha'}")
            print(f"DEBUG: Output path: {output_dir / 'data' / first_uid / 'img.nii.gz'}")
            print(f"DEBUG: Input file exists: {(image_dir / f'{first_uid}.mha').exists()}")
            
            result = convert_case(first_uid, image_dir, output_dir)
            if result:
                print(f"DEBUG: Error: {result}")
            else:
                print("DEBUG: Success!")
                # Verify the output file was created
                output_path = output_dir / 'data' / first_uid / 'img.nii.gz'
                print(f"DEBUG: Output file exists: {output_path.exists()}")
                if output_path.exists():
                    print(f"DEBUG: Output file size: {output_path.stat().st_size} bytes")
        else:
            print("DEBUG: No SeriesInstanceUIDs found in CSV")
        exit()
    
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
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting MHA to NIfTI (Phase 1 Parallel)"):
                result = future.result()
                if result:
                    errors.append(result)
    
    # Print results summary
    total_series = len(unique_uids)
    missing_files = len([e for e in errors if "does not exist" in e])
    successful_series = total_series - missing_files
    
    print(f"\nPhase 1 Conversion Complete!")
    print(f"Total series in CSV: {total_series}")
    print(f"Missing .mha files: {missing_files}")
    print(f"Successfully converted: {successful_series}")
    print(f"Other errors: {len(errors) - missing_files}")
    
    if errors:
        print(f"\nFirst 10 conversion errors:")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    print(f"\nPhase 1 files saved to: {output_dir}/data/[SeriesInstanceUID]/img.nii.gz")
    
    # Show example of created files
    if successful_series > 0:
        data_dir = output_dir / "data"
        if data_dir.exists():
            created_dirs = list(data_dir.glob("*"))[:3]
            print(f"\nExample created directories:")
            for d in created_dirs:
                img_file = d / "img.nii.gz"
                status = "✓" if img_file.exists() else "✗"
                print(f"  {status} {img_file}")
            if len(created_dirs) > 3:
                print(f"  ... and {len(list(data_dir.glob('*'))) - 3} more directories")
   
# import pandas as pd
# import SimpleITK as sitk
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm import tqdm
# import argparse


# def convert_case(uid, image_dir, output_dir):
#     input_path = image_dir / f"{uid}.mha"
#     output_dir = output_dir / uid
#     output_dir.mkdir(parents=True, exist_ok=True)
#     output_path = output_dir/ f"img.nii.gz"

#     if not input_path.exists():
#         return f"Warning: {input_path} does not exist."

#     try:
#         image = sitk.ReadImage(str(input_path))
#         sitk.WriteImage(image, str(output_path))
#         return None
#     except Exception as e:
#         return f"Failed to convert {uid}: {e}"


# ### python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/luna25/step1_mha_to_nifti.py --num_workers 20 

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--csv_path', default='/radraid2/mwahianwar/miccai25/luna25challenge/data/LUNA25_Public_Training_Development_Data.csv', type=str)
#     parser.add_argument('--image_dir', default='/radraid2/mwahianwar/miccai25/luna25challenge/data/luna25_images', type=str)
#     parser.add_argument('--output_dir', default='/radraid2/mwahianwar/MST/luna25/preprocessed', type=str)
#     parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel processes (only used if not sequential)")
#     parser.add_argument('--sequential', action='store_true', help="Run conversion sequentially instead of in parallel")

#     args = parser.parse_args()

#     csv_path = Path(args.csv_path)
#     image_dir = Path(args.image_dir)
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(csv_path)
#     unique_uids = df["SeriesInstanceUID"].dropna().unique()

#     errors = []

#     if args.sequential:
#         for uid in tqdm(unique_uids, desc="Sequentially converting MHA to NIfTI"):
#             result = convert_case(uid, image_dir, output_dir)
#             if result:
#                 errors.append(result)
#     else:
#         with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
#             futures = {
#                 executor.submit(convert_case, uid, image_dir, output_dir): uid for uid in unique_uids
#             }
#             for future in tqdm(as_completed(futures), total=len(futures), desc="Converting MHA to NIfTI (Parallel)"):
#                 result = future.result()
#                 if result:
#                     errors.append(result)

#     if errors:
#         print("\nSome conversions failed:")
#         for e in errors:
#             print(e)
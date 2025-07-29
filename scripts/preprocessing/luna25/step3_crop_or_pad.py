from pathlib import Path
import SimpleITK as sitk
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Global variable to store the SNGAN lookup table
sngan_lookup = None

def load_sngan_lookup(lookup_csv_path):
    """Load the SNGAN lookup table CSV"""
    global sngan_lookup
    try:
        lookup_df = pd.read_csv(lookup_csv_path)
        # Create a dictionary mapping from key (SeriesInstanceUID) to SNGAN path
        sngan_lookup = {}
        for _, row in lookup_df.iterrows():
            key = row['key']
            sngan_path = row['SNGAN']
            if pd.notna(sngan_path) and sngan_path.strip():  # Only add non-empty SNGAN paths
                sngan_lookup[key] = sngan_path
        print(f"Loaded SNGAN lookup table with {len(sngan_lookup)} entries")
        return True
    except Exception as e:
        print(f"Error loading SNGAN lookup table: {e}")
        return False

def get_sngan_image_path(uid):
    """Get the SNGAN preprocessed image path for a given SeriesInstanceUID"""
    global sngan_lookup
    if sngan_lookup and uid in sngan_lookup:
        sngan_path = sngan_lookup[uid]
        # Convert to Path object
        return Path(sngan_path)
    return None

def align_image_to_reference(image, reference_image):
    """
    Align SNGAN image to match the reference image orientation.
    Based on logs and visual misalignment: Y-axis flip needed.
    """
    try:
        print("=== Performing SNGAN Alignment ===")
        
        # Convert to numpy array
        img_array = sitk.GetArrayFromImage(image)  # Z,Y,X
        print(f"Original SNGAN shape: {img_array.shape}")
        
        img_array_transposed = np.transpose(img_array, (0, 2, 1))
        print(f"After transpose Y,X: {img_array_transposed.shape}")
        
        # Step 2: Flip X axis (axis=2 after transpose)
        img_array_aligned = np.flip(img_array_transposed, axis=2)
        print(f"After flip X: {img_array_aligned.shape}")
        
        
        # Create new image with aligned data
        aligned_image = sitk.GetImageFromArray(img_array_aligned)
        
        # Copy spatial properties from reference
        aligned_image.SetOrigin(reference_image.GetOrigin())
        aligned_image.SetSpacing(reference_image.GetSpacing())
        aligned_image.SetDirection(reference_image.GetDirection())
        
        return aligned_image
        
    except Exception as e:
        print(f"Warning: Could not align SNGAN image: {e}")
        return image

def crop_lesion_pair(uid, lesion_id, lesion_coords, image_dir, output_dir, crop_shape, padding_value):
    """Crop both image and segmentation mask for a single lesion using SimpleITK"""
    
    # Segmentation path (always from original preprocessed data)
    seg_path = image_dir / "data" / uid / f"seg_{lesion_id}.nii.gz"
    
    # Try to get SNGAN preprocessed image path first
    sngan_img_path = get_sngan_image_path(uid)
    use_sngan = False
    
    if sngan_img_path and sngan_img_path.exists():
        # Use SNGAN preprocessed image
        img_path = sngan_img_path
        use_sngan = True
        print(f"Using SNGAN preprocessed image for {uid}: {img_path}")
    else:
        # Fallback to original preprocessed image path
        img_path = image_dir / "data" / uid / "img.nii.gz"
        if sngan_img_path:
            print(f"Warning: SNGAN path not found for {uid}, falling back to: {img_path}")
    
    # Check if both files exist
    if not img_path.exists():
        return f"Missing image: {img_path}"
    
    if not seg_path.exists():
        return f"Missing segmentation: {seg_path}"

    try:
        # Load segmentation first (this is our reference space)
        segmentation = sitk.ReadImage(str(seg_path))
        
        # Load image
        image = sitk.ReadImage(str(img_path))
        
        # If using SNGAN image, align it to the segmentation space
        if use_sngan:
            # Load the original reference image to check for spatial differences
            ref_img_path = image_dir / "data" / uid / "img.nii.gz"
            if ref_img_path.exists():
                try:
                    reference_image = sitk.ReadImage(str(ref_img_path))
                    print("ref_shape", reference_image.GetSize())
                    
                    # Check if SNGAN image has different spatial properties
                    if (image.GetOrigin() != reference_image.GetOrigin() or 
                        image.GetSpacing() != reference_image.GetSpacing() or 
                        image.GetDirection() != reference_image.GetDirection() or
                        image.GetSize() != reference_image.GetSize()):
                        
                        print(f"Aligning SNGAN image to reference space for {uid}")
                        image = align_image_to_reference(image, reference_image)

                    print("image_shape", image.GetSize())
                        
                except Exception as e:
                    print(f"Warning: Could not load reference image for alignment: {e}")
        
    except Exception as e:
        return f"Failed to load files for {uid}_{lesion_id}: {e}"

    # Get physical coordinates
    x, y, z = lesion_coords["CoordX"], lesion_coords["CoordY"], lesion_coords["CoordZ"]
    physical_point = [float(x), float(y), float(z)]

    # Transform to voxel coordinates using SimpleITK
    try:
        voxel_index = image.TransformPhysicalPointToIndex(physical_point)
        x_vox, y_vox, z_vox = voxel_index
    except RuntimeError:
        return f"Physical point {physical_point} outside image bounds for {uid} lesion {lesion_id}"

    # Get image size
    size = image.GetSize()  # (width, height, depth) = (x, y, z)
    
    # Calculate crop boundaries in SimpleITK coordinates (x, y, z)
    half_crop_x, half_crop_y, half_crop_z = crop_shape[0]//2, crop_shape[1]//2, crop_shape[2]//2
    
    start_x = max(0, x_vox - half_crop_x)
    end_x = min(size[0], x_vox + half_crop_x)
    start_y = max(0, y_vox - half_crop_y)
    end_y = min(size[1], y_vox + half_crop_y)
    start_z = max(0, z_vox - half_crop_z)
    end_z = min(size[2], z_vox + half_crop_z)
    
    # Crop using SimpleITK
    try:
        # Extract region from both image and segmentation
        crop_filter = sitk.ExtractImageFilter()
        
        # Set the extraction region [start_x, start_y, start_z] and size
        crop_size = [end_x - start_x, end_y - start_y, end_z - start_z]
        crop_index = [start_x, start_y, start_z]
        
        crop_filter.SetSize(crop_size)
        crop_filter.SetIndex(crop_index)
        
        cropped_img = crop_filter.Execute(image)
        cropped_seg = crop_filter.Execute(segmentation)
        
        # Get actual cropped size
        actual_size = cropped_img.GetSize()
        
        # Pad if necessary to reach target crop_shape
        target_size = [crop_shape[0], crop_shape[1], crop_shape[2]]  # [X, Y, Z] for SimpleITK
        
        if actual_size != target_size:
            # Calculate padding needed
            pad_needed = [target_size[i] - actual_size[i] for i in range(3)]
            
            if any(p > 0 for p in pad_needed):
                # Pad using SimpleITK
                pad_filter = sitk.ConstantPadImageFilter()
                
                # Calculate padding before and after for each dimension
                pad_lower = [p // 2 for p in pad_needed]
                pad_upper = [p - p//2 for p in pad_needed]
                
                pad_filter.SetPadLowerBound(pad_lower)
                pad_filter.SetPadUpperBound(pad_upper)
                pad_filter.SetConstant(float(padding_value))
                
                # Pad image with padding_value
                cropped_img = pad_filter.Execute(cropped_img)
                
                # Pad segmentation with 0 (background)
                pad_filter.SetConstant(0.0)
                cropped_seg = pad_filter.Execute(cropped_seg)
        
        # Create output directory
        out_dir = output_dir / uid
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths
        out_img_path = out_dir / f"img_{lesion_id}.nii.gz"
        out_seg_path = out_dir / f"seg_{lesion_id}.nii.gz"
        
        # Save cropped images
        sitk.WriteImage(cropped_img, str(out_img_path))
        sitk.WriteImage(cropped_seg, str(out_seg_path))
        
        return None  # Success
        
    except Exception as e:
        return f"Failed to crop/save {uid}_{lesion_id} at ({x},{y},{z}): {e}"

def crop_case(uid_df_tuple):
    """Process all lesions for a single SeriesInstanceUID"""
    uid, rows = uid_df_tuple
    errors = []

    for idx, row in rows.iterrows():
        lesion_id = row.get("LesionID", idx)
        
        # Process each lesion individually
        error = crop_lesion_pair(
            uid=uid,
            lesion_id=lesion_id,
            lesion_coords=row,
            image_dir=image_dir,
            output_dir=output_dir,
            crop_shape=crop_shape,
            padding_value=padding_value
        )
        
        if error:
            errors.append(error)

    return errors if errors else None

### Usage:
# python step3_crop_simpleitk.py --sequential 
# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/luna25/step3_crop_or_pad.py --num_workers 20 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop both image and segmentation files using SimpleITK with SNGAN preprocessed images")
    parser.add_argument('--csv_path', default='/radraid2/mwahianwar/miccai25/luna25challenge/data/LUNA25_Public_Training_Development_Data.csv', type=str)
    parser.add_argument('--preprocessed_image_dir', default='/radraid2/mwahianwar/MST/luna25/preprocessed', type=str)
    parser.add_argument('--output_dir', default='/radraid2/mwahianwar/MST/luna25/preprocessed_crop_sngan', type=str)
    parser.add_argument('--sngan_lookup_csv', default='/cvib2/apps/personal/wasil/lib/classification/MST/ctnorm_lookup.csv', type=str, help="Path to the SNGAN lookup CSV file")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel workers")
    parser.add_argument('--sequential', action='store_true', help="Run sequentially instead of parallel")

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    image_dir = Path(args.preprocessed_image_dir)
    output_dir = Path(args.output_dir)
    sngan_lookup_csv = Path(args.sngan_lookup_csv)
    
    # Load SNGAN lookup table
    if not load_sngan_lookup(sngan_lookup_csv):
        print(f"ERROR: Could not load SNGAN lookup table from: {sngan_lookup_csv}")
        exit(1)
    
    # Check permissions and create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"ERROR: No permission to create directory: {output_dir}")
        print(f"Try running with: --output_dir ./preprocessed_crop_sngan")
        exit(1)

    # Read and validate CSV
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["SeriesInstanceUID", "CoordX", "CoordY", "CoordZ", "LesionID"])
        grouped = list(df.groupby("SeriesInstanceUID"))
    except Exception as e:
        print(f"ERROR: Could not read CSV file: {csv_path}")
        print(f"Error: {e}")
        exit(1)

    crop_shape = (256, 256, 32)  # (x, y, z)
    padding_value = -1024

    print(f"SimpleITK Cropping with SNGAN Complete Alignment:")
    print(f"Total series to process: {len(grouped)}")
    print(f"Total lesions to process: {len(df)}")
    print(f"SNGAN lookup entries: {len(sngan_lookup) if sngan_lookup else 0}")
    print(f"Crop shape (X,Y,Z): {crop_shape}")
    print(f"Input directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"SNGAN lookup CSV: {sngan_lookup_csv}")
    print(f"Complete alignment: ENABLED (axis ordering + spatial metadata)")

    # Check if input directory structure exists
    data_dir = image_dir / "data"
    if not data_dir.exists():
        print(f"ERROR: Input data directory does not exist: {data_dir}")
        print(f"Make sure step1 and step2 completed successfully.")
        exit(1)

    # Count available files
    available_images = list(data_dir.glob("*/img.nii.gz"))
    available_segs = list(data_dir.glob("*/seg_*.nii.gz"))
    print(f"Found {len(available_images)} original images and {len(available_segs)} segmentations")

    errors = []

    if args.sequential:
        for uid_df in tqdm(grouped, desc="Cropping pairs with SNGAN Complete Alignment (Sequential)"):
            err = crop_case(uid_df)
            if err:
                errors.extend(err if isinstance(err, list) else [err])
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(crop_case, uid_df): uid_df[0] for uid_df in grouped}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Cropping pairs with SNGAN Complete Alignment (Parallel)"):
                result = future.result()
                if result:
                    errors.extend(result if isinstance(result, list) else [result])

    # Print detailed summary
    total_series = len(grouped)
    total_lesions = len(df)
    missing_images = len([e for e in errors if "Missing image" in e])
    missing_segs = len([e for e in errors if "Missing segmentation" in e])
    out_of_bounds = len([e for e in errors if "outside image bounds" in e])
    other_errors = len(errors) - missing_images - missing_segs - out_of_bounds
    successful_lesions = total_lesions - len(errors)
    
    print(f"\nSimpleITK Cropping with SNGAN Complete Alignment Complete:")
    print(f"Total series: {total_series}")
    print(f"Total lesions processed: {total_lesions}")
    print(f"Missing images: {missing_images}")
    print(f"Missing segmentations: {missing_segs}")
    print(f"Out of bounds: {out_of_bounds}")
    print(f"Other errors: {other_errors}")
    print(f"Successfully cropped pairs: {successful_lesions}")
    print(f"Success rate: {successful_lesions/total_lesions*100:.1f}%")

    if errors and len(errors) <= 20:
        print(f"\nErrors:")
        for e in errors:
            print(f"  {e}")
    elif len(errors) > 20:
        print(f"\nFirst 10 errors:")
        for e in errors[:10]:
            print(f"  {e}")
        print(f"  ... and {len(errors) - 10} more errors")
    
    print(f"\nCropped files saved to:")
    print(f"  Images: {output_dir}/[SeriesInstanceUID]/img_[LesionID].nii.gz")
    print(f"  Segmentations: {output_dir}/[SeriesInstanceUID]/seg_[LesionID].nii.gz")
    
    # Show example of created files if successful
    if successful_lesions > 0:
        created_dirs = list(output_dir.glob("*"))[:3]
        print(f"\nExample created files:")
        for d in created_dirs:
            if d.is_dir():
                img_files = list(d.glob("img_*.nii.gz"))[:2]
                seg_files = list(d.glob("seg_*.nii.gz"))[:2]
                for img_f, seg_f in zip(img_files, seg_files):
                    print(f"  ✓ {img_f}")
                    print(f"  ✓ {seg_f}")
                if len(list(d.glob("img_*.nii.gz"))) > 2:
                    total_pairs = len(list(d.glob("img_*.nii.gz")))
                    print(f"    ... and {total_pairs - 2} more pairs in {d.name}")
        if len(created_dirs) > 3:
            print(f"  ... and {len(created_dirs) - 3} more series")
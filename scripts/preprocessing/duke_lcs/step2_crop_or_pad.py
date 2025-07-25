from pathlib import Path
import SimpleITK as sitk
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


def crop_lesion_pair(uid, lesion_id, lesion_coords, image_dir, output_dir, crop_shape, padding_value):
    """Crop both image and segmentation mask for a single lesion"""
    img_path = image_dir / uid / "img.nii.gz"
    seg_path = image_dir / uid / f"{lesion_id}.nii.gz"

    if not img_path.exists():
        return f"Missing image: {img_path}"
    if not seg_path.exists():
        return f"Missing segmentation: {seg_path}"

    try:
        image = sitk.ReadImage(str(img_path))
        segmentation = sitk.ReadImage(str(seg_path))
    except Exception as e:
        return f"Failed to load files for {uid}_{lesion_id}: {e}"

    # Get physical coordinates - handle both naming conventions
    x = lesion_coords.get("coordX") 
    y = lesion_coords.get("coordY") 
    z = lesion_coords.get("coordZ") 
    
    physical_point = [float(x), float(y), float(z)]

    # Transform to voxel coordinates
    try:
        x_vox, y_vox, z_vox = image.TransformPhysicalPointToIndex(physical_point)
    except RuntimeError:
        return f"Physical point outside bounds for {uid} lesion {lesion_id}"

    # Calculate crop boundaries
    size = image.GetSize()
    half_crop_x, half_crop_y, half_crop_z = crop_shape[0]//2, crop_shape[1]//2, crop_shape[2]//2
    
    start_x = max(0, x_vox - half_crop_x)
    end_x = min(size[0], x_vox + half_crop_x)
    start_y = max(0, y_vox - half_crop_y)
    end_y = min(size[1], y_vox + half_crop_y)
    start_z = max(0, z_vox - half_crop_z)
    end_z = min(size[2], z_vox + half_crop_z)
    
    try:
        # Crop both image and segmentation
        crop_filter = sitk.ExtractImageFilter()
        crop_filter.SetSize([end_x - start_x, end_y - start_y, end_z - start_z])
        crop_filter.SetIndex([start_x, start_y, start_z])
        
        cropped_img = crop_filter.Execute(image)
        cropped_seg = crop_filter.Execute(segmentation)
        
        # Pad if necessary
        actual_size = cropped_img.GetSize()
        target_size = list(crop_shape)
        
        if actual_size != target_size:
            pad_needed = [target_size[i] - actual_size[i] for i in range(3)]
            if any(p > 0 for p in pad_needed):
                pad_filter = sitk.ConstantPadImageFilter()
                pad_lower = [p // 2 for p in pad_needed]
                pad_upper = [p - p//2 for p in pad_needed]
                pad_filter.SetPadLowerBound(pad_lower)
                pad_filter.SetPadUpperBound(pad_upper)
                
                pad_filter.SetConstant(float(padding_value))
                cropped_img = pad_filter.Execute(cropped_img)
                
                pad_filter.SetConstant(0.0)
                cropped_seg = pad_filter.Execute(cropped_seg)
        
        # Save cropped images
        out_dir = output_dir / uid
        out_dir.mkdir(parents=True, exist_ok=True)
        
        sitk.WriteImage(cropped_img, str(out_dir / f"img_{lesion_id}.nii.gz"))
        sitk.WriteImage(cropped_seg, str(out_dir / f"seg_{lesion_id}.nii.gz"))

        return None
        
    except Exception as e:
        return f"Failed to crop/save {uid}_{lesion_id}: {e}"


def crop_case(uid_df_tuple, image_dir, output_dir, crop_shape, padding_value):
    """Process all lesions for a single patient"""
    uid, rows = uid_df_tuple
    errors = []

    for idx, row in rows.iterrows():
        lesion_id = row.get("nodule_id", idx)
        error = crop_lesion_pair(uid, lesion_id, row, image_dir, output_dir, crop_shape, padding_value)
        if error:
            errors.append(error)

    return errors if errors else None

### 
### python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/duke_lcs/step2_crop_or_pad.py --num_workers 20 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/radraid2/mwahianwar/duke_lcs/data/DLCSD24_Annotations.csv', type=str)
    parser.add_argument('--preprocessed_image_dir', default='/radraid2/mwahianwar/MST/duke_lcs/preprocessed', type=str)
    parser.add_argument('--output_dir', default='/radraid2/mwahianwar/MST/duke_lcs/preprocessed_crop', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--sequential', action='store_true')

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    image_dir = Path(args.preprocessed_image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read and validate CSV
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["patient-id", "coordX", "coordY", "coordZ", "nodule_id"])
    grouped = list(df.groupby("patient-id"))

    crop_shape = (256, 256, 32)
    padding_value = -1024

    print(f"Processing {len(grouped)} patients, {len(df)} lesions")
    print(f"Crop shape: {crop_shape}, Output: {output_dir}")

    errors = []

    if args.sequential:
        for uid_df in tqdm(grouped, desc="Cropping"):
            err = crop_case(uid_df, image_dir, output_dir, crop_shape, padding_value)
            if err:
                errors.extend(err)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(crop_case, uid_df, image_dir, output_dir, crop_shape, padding_value): uid_df[0] 
                      for uid_df in grouped}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Cropping"):
                result = future.result()
                if result:
                    errors.extend(result)

    # Results
    successful = len(df) - len(errors)
    print(f"Success: {successful}/{len(df)} ({successful/len(df)*100:.1f}%)")
    
    if errors and len(errors) <= 10:
        print("Errors:")
        for e in errors:
            print(f"  {e}")
    elif len(errors) > 10:
        print(f"First 5 errors:")
        for e in errors[:5]:
            print(f"  {e}")
        print(f"  ... and {len(errors) - 5} more")
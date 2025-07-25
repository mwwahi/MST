import pandas as pd
import numpy as np
import torchio as tio
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

def create_mask_for_lesion(row, path_root_data):
    """Create a mask for a single lesion"""
    uid = row["SeriesInstanceUID"]
    lesion_id = row.get("LesionID", row.name)
    
    img_path = path_root_data / "data" / uid / "img.nii.gz"
    
    if not img_path.exists():
        return f"Image not found: {img_path}"

    try:
        # Load with SimpleITK
        sitk_img = sitk.ReadImage(str(img_path))
        spacing = sitk_img.GetSpacing()
        origin = sitk_img.GetOrigin()
        direction = sitk_img.GetDirection()
        
        # Transform physical coordinates to voxel index
        physical_point = [float(row["CoordX"]), float(row["CoordY"]), float(row["CoordZ"])]
        try:
            voxel_index = sitk_img.TransformPhysicalPointToIndex(physical_point)
            x_vox, y_vox, z_vox = voxel_index
        except RuntimeError:
            return f"Point outside bounds: {uid} lesion {lesion_id}"

        # Initialize a blank mask using SimpleITK
        mask_array = np.zeros(sitk_img.GetSize()[::-1], dtype=np.uint8)  # Note shape reversal
        mask_array[z_vox, y_vox, x_vox] = 1
        
        sitk_mask = sitk.GetImageFromArray(mask_array)
        sitk_mask.SetSpacing(spacing)
        sitk_mask.SetOrigin(origin)
        sitk_mask.SetDirection(direction)

        out_path = path_root_data / "data" / uid / f'seg_{lesion_id}.nii.gz'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk_mask, str(out_path))
        
        return None

    except Exception as e:
        return f"Error: {uid} lesion {lesion_id}: {str(e)}"


def process_series(uid_df_tuple, path_root_data):
    """Process all lesions for a single series"""
    uid, rows = uid_df_tuple
    errors = []
    
    for idx, row in rows.iterrows():
        error = create_mask_for_lesion(row, path_root_data)
        if error:
            errors.append(error)
    
    return errors if errors else None

def main(csv_path, path_root_data, args):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["SeriesInstanceUID", "CoordX", "CoordY", "CoordZ"])
    
    data_dir = path_root_data / "data"
    available_images = list(data_dir.glob("*/img.nii.gz"))
    
    print(f"Processing {len(df)} lesions from {len(available_images)} images...")
    
    errors = []
    
    if args.sequential:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating masks"):
            error = create_mask_for_lesion(row, path_root_data)
            if error:
                errors.append(error)
    else:
        grouped = list(df.groupby("SeriesInstanceUID"))
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_series, uid_df, path_root_data): uid_df[0] 
                for uid_df in grouped
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Creating masks"):
                result = future.result()
                if result:
                    errors.extend(result if isinstance(result, list) else [result])
    
    successful_masks = len(df) - len(errors)
    print(f"Created {successful_masks} masks ({successful_masks/len(df)*100:.1f}% success rate)")
    
    if errors and len(errors) <= 20:
        print("Errors:")
        for e in errors:
            print(f"  {e}")

### Usage:
# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/luna25/step2_make_mask.py --num_workers 20 
# python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/luna25/step2_make_mask.py --sequential 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/radraid2/mwahianwar/miccai25/luna25challenge/data/LUNA25_Public_Training_Development_Data.csv', type=str)
    parser.add_argument('--output_dir', default='/radraid2/mwahianwar/MST/luna25/preprocessed', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--sequential', action='store_true')
    
    args = parser.parse_args()
    main(Path(args.csv_path), Path(args.output_dir), args)


# import pandas as pd
# import numpy as np
# import torchio as tio
# from pathlib import Path

# # Paths
# # csv_path = Path("/radraid2/mwahianwar/duke_lcs/data/DLCSD24_Annotations.csv")
# # path_root_data = Path("/radraid2/mwahianwar/duke_lcs/data/")
# # path_root_download = Path("/radraid2/mwahianwar/duke_lcs/download/")  # used to mimic relative scan path
# def main(csv_path, path_root_data, ):
#     # Load annotations
#     df = pd.read_csv(csv_path)

#     for idx, row in df.iterrows():
#         nifti_path = Path(row["ct_nifti_file"])

#         idx
#         # try:
#         #     path_rel = nifti_path.parent.relative_to(path_root_download)  # mimic get_path_to_dicom_files
#         # except ValueError:
#         #     print(f"Skipping index {idx}: {nifti_path} not under path_root_download")
#         #     continue

#         img_path = path_root_data / path_rel / "img.nii.gz"
#         if not img_path.exists():
#             print(f"Image not found at: {img_path}")
#             continue

#         # Load volume
#         vol = tio.ScalarImage(img_path)

#         # Convert physical coords to voxel coords
#         x,y,z = row["CoordX"], row["CoordY"], row["CoordZ"]
#         physical_coords = np.array([x,y,z], dtype=np.float32)
#         affine_inv = np.linalg.inv(vol.affine)
#         voxel_coords = affine_inv @ np.append(physical_coords, 1)
#         voxel_coords = np.round(voxel_coords[:3]).astype(int)  # z, y, x

#         z, y, x = voxel_coords
#         if not (0 <= z < vol.shape[1] and 0 <= y < vol.shape[2] and 0 <= x < vol.shape[3]):
#             print(f"Out of bounds at index {idx}: voxel {voxel_coords} in shape {vol.shape[1:]}")
#             continue

#         # Create empty mask
#         mask_vol = np.zeros(vol.spatial_shape, dtype=np.uint8)
#         mask_vol[z, y, x] = 1

#         # Wrap and save
#         mask_vol = tio.LabelMap(tensor=mask_vol[None], affine=vol.affine)
#         out_path = path_root_data / path_rel / f'seg_{idx}.nii.gz'
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         mask_vol.save(out_path)



# ### python /cvib2/apps/personal/wasil/lib/classification/MST/scripts/preprocessing/luna25/step1_mha_to_nifti.py --num_workers 20 
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--csv_path', default='/radraid2/mwahianwar/miccai25/luna25challenge/data/LUNA25_Public_Training_Development_Data.csv', type=str)
#     parser.add_argument('--image_dir', default='/radraid2/mwahianwar/miccai25/luna25challenge/data/luna25_images', type=str)
#     parser.add_argument('--output_dir', default='/radraid2/mwahianwar/MST/luna25/preprocessed', type=str)
#     parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel processes (only used if not sequential)")
#     parser.add_argument('--sequential', action='store_true', help="Run conversion sequentially instead of in parallel")
    
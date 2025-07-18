

from pathlib import Path 
import logging 
import sys
import pandas as pd 
import pydicom.datadict
import pydicom.dataelem
import pydicom.sequence
import pydicom.valuerep
from tqdm import tqdm
import torchio as tio 
import torch 
import pydicom
import pylidc as pl
from multiprocessing import Pool


# pl.config["dicom"] = {
#     "path": "/radraid2/mwahianwar/LIDC/download/TCIA_LIDC-IDRI_20200921/LIDC-IDRI",
#     "warn": True,
# }
from pathlib import Path

# Override the method at runtime
def patched_get_path_to_dicom_files(self):
    # base_path = "/radraid2/mwahianwar/LIDC/data/TCIA_LIDC-IDRI_20200921/LIDC-IDRI"
    base_path = "/radraid2/mwahianwar/LIDC/data/LIDC-IDRI-pylidc"
    return str(Path(base_path) / self.patient_id)

pl.Scan.get_path_to_dicom_files = patched_get_path_to_dicom_files



def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        # return [maybe_convert(item) for item in x]
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.dataset.Dataset):  
        # return dataset2dict(x)
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x 


# def dataset2dict(ds, exclude=['PixelData', '']):
#     return {keyword:value for key in ds.keys() 
#             if ((keyword := ds[key].keyword) not in exclude)  and ((value := maybe_convert(ds[key].value)) is not None) }
def dataset2dict(ds, exclude=['PixelData', '']):
    out = {}
    for key in ds.keys():
        keyword = ds[key].keyword
        if keyword in exclude:
            continue
        value = maybe_convert(ds[key].value)
        if value is not None:
            out[keyword] = value
    return out

# def scan2nifti(scan_id):
#     print(scan_id)
#     scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()

#     # # Get path to series
#     # path_series = Path(scan.get_path_to_dicom_files())
def get_series_uids_in_dir(path):
    uids = set()
    for dcm_path in Path(path).glob("*.dcm"):
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            uids.add(ds.SeriesInstanceUID)
        except Exception as e:
            print(f"Failed to read {dcm_path}: {e}")
    return uids

def scan2nifti(scan_id):
    print(scan_id)
    scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
    if scan is None:
        logger.warning(f"Scan ID {scan_id} not found.")
        return None

    path_series = Path(scan.get_path_to_dicom_files())
    print(path_series)
    if not path_series.exists():
        logger.warning(f"Scan path not found: {path_series}")
        return None
    # Try loading images first
    try:
        images = scan.load_all_dicom_images()
        if not images:

            print("Expected UID:", scan.series_instance_uid)
            print("UIDs found in folder:", get_series_uids_in_dir(path_series))            
            logger.warning(f"No DICOM slices found for {scan.patient_id} at {path_series}")
            return None
    except Exception as e:
        logger.warning(f"Failed to load DICOM slices for {scan.patient_id}: {e}")
        return None

    # Proceed to volume creation
    try:
        img_pl = scan.to_volume()
    except Exception as e:
        logger.warning(f"Failed to convert to volume for {scan.patient_id}: {e}")
        return None

    # Read DICOM
    # img_pl = scan.to_volume()
    affine = torch.zeros((4,4))
    affine[0, 0] = scan.spacings[0]
    affine[1, 1] = scan.spacings[1]
    affine[2, 2] = scan.spacings[2]
    img_tio = tio.ScalarImage(tensor=img_pl[None], affine=affine)

    # Read Metadata 
    ds = pydicom.dcmread(next(path_series.glob('*.dcm'), None), stop_before_pixels=True)
    metadata = dataset2dict(ds)
    
    # Create output folder 
    rel_path = path_series.relative_to(path_root_in)
    path_out_dir = path_root_out_data/rel_path
    path_out_dir.mkdir(exist_ok=True, parents=True)

    # Write 
    filename = 'img.nii.gz' #metadata['ProtocolName']
    logger.info(f"Writing file: {filename}:")
    img_tio.save(path_out_dir/filename )

    # Add additional information 
    metadata['_SpatialShape'] = list(img_tio.spatial_shape)
    metadata['_Path'] = str(rel_path/filename)

    return metadata





if __name__ == "__main__":
    # WARNING: DON'T try to read DICOM yourself - LIDC is messy  :( 
    # Use pylidc, as it "fixes", for example "Some scans contain multiple slices with the same `z` coordinate" - wtf  
    # Follow instructions: https://pylidc.github.io/install.html

    # Setting 
    # path_root = Path('/home/gustav/Coscine_Public/LIDC-IDRI')
    path_root = Path('/radraid2/mwahianwar/LIDC')
    # path_root_in = path_root/'data/TCIA_LIDC-IDRI_20200921/LIDC-IDRI'
    path_root_in = path_root/'data/LIDC-IDRI-pylidc'
    path_root_out = path_root/'preprocessed'
    path_root_out_data = path_root_out/'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)


    # Logging 
    path_log_file = path_root_out/'preprocessing.log'
    logger = logging.getLogger(__name__)
    s_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(path_log_file, 'w')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[s_handler, f_handler])
    

    # Get all scans 
    scan_ids = range(1, len(list(pl.query(pl.Scan)))+1)
    # scan_ids = [136, 316, 333, 357, 447, 489, 568, 843, 844, 845, 846, 847, 848, 849, 851, 945]
    # scan_ids = [136, 316, 447, 489]
    scan_ids = [9,10]
    # scan_ids = [7, 315, 334, 358, 446, 490]
    # Option 1: Multi-CPU 
    # metadata_list = []
    # with Pool() as pool:
    #     for meta in tqdm(pool.imap_unordered(scan2nifti, scan_ids), total=len(scan_ids)):
    #         metadata_list.append(meta)

    # Option 2: Single-CPU (if you need a coffee break)
    metadata_list = []
    failed_ids = []
    for scan_id in tqdm(scan_ids):
        meta = scan2nifti(scan_id)
        if meta is not None:
            metadata_list.append(meta)
        else:
            failed_ids.append(scan_id)

    # Check export 
    path_exports = [path.relative_to(path_root_out) for path in path_root_out.rglob('img.nii.gz')]
    num_patients = list(set([path.parts[0] for path in  path_exports]))
    print("Exported Patients:", len(num_patients), " of 1010")
    print("Exported Studies:", len(path_exports), " of 1018 (pylidc) or 1308 (TCIA)")


    # failed_ids = []
    # for scan_id in tqdm(scan_ids):
    #     meta = scan2nifti(scan_id)
    #     if meta is not None:
    #         metadata_list.append(meta)
    #     else:
    #         failed_ids.append(scan_id)

    logger.info(f"Failed scan IDs: {failed_ids}")


    df = pd.DataFrame([m for m in metadata_list if m is not None])
    
    # # Save metadata 
    # df = pd.DataFrame(metadata_list)
    # df.to_csv(path_root_out/'metadata_final_2.csv', index=False)
    
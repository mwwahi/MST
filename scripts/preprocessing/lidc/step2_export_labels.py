

from pathlib import Path 
import pylidc as pl
import numpy as np 
import torchio as tio 
import pandas as pd 
from tqdm import tqdm
from multiprocessing import Pool, Manager
from pylidc.utils import consensus


# Override the method at runtime
def patched_get_path_to_dicom_files(self):
    # base_path = "/radraid2/mwahianwar/LIDC/data/TCIA_LIDC-IDRI_20200921/LIDC-IDRI"
    base_path = "/radraid2/mwahianwar/LIDC/data/LIDC-IDRI-pylidc"
    return str(Path(base_path) / self.patient_id)

from sklearn.cluster import DBSCAN
import numpy as np

def manual_cluster_annotations(scan, max_dist_mm=5):
    anns = scan.annotations
    if len(anns) == 0:
        return []

    # Get centroids and convert to physical space (mm)
    centroids_px = np.array([ann.centroid for ann in anns])  # (z, y, x)
    print(centroids_px, "centroids_px")
    print(scan.slice_thickness, "slice thickness")
    print(scan.pixel_spacing, "pixel spacing")
    spacing = np.array([scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing], dtype=float)  # (z, y, x)
    centroids_mm = centroids_px * spacing  # convert to mm space

    # Cluster using DBSCAN
    db = DBSCAN(eps=max_dist_mm, min_samples=1)
    labels = db.fit_predict(centroids_mm)

    # Group annotations by cluster ID
    clusters = []
    for lbl in np.unique(labels):
        cluster = [ann for ann, l in zip(anns, labels) if l == lbl]
        clusters.append(cluster)

    return clusters

pl.Scan.get_path_to_dicom_files = patched_get_path_to_dicom_files

LABELS = ['subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation',
          'texture', 'malignancy']


def scan2labels(scan_id):
    scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
    
    # Read nifti (required for correct affine matrix)
    path_rel = Path(scan.get_path_to_dicom_files()).relative_to(path_root_download)
    vol = tio.ScalarImage(path_root_data/path_rel/'img.nii.gz') 
    print(scan_id, path_root_data/path_rel/'img.nii.gz')

    scan_ann = []

    try:
        clusters = scan.cluster_annotations()
    # except pl.Scan.ClusterError:
    except:
        print(f"[!] Using manual clustering for {scan.patient_id}")
        clusters = manual_cluster_annotations(scan, max_dist_mm=5)  # adjust threshold if needed

    for nod_idx, nodules in enumerate(clusters): # Each scan has multiple nodules
    # for nod_idx, nodules in enumerate(scan.cluster_annotations()): # Each scan has multiple nodules
        for ann_idx, ann in enumerate(nodules): # Each nodule was rated between 1 and 4 raters
            ann_dict = {label:getattr(ann, label) for label in LABELS}
            ann_dict['bbox'] = [[d.start, d.stop] for d in  ann.bbox()]
            ann_dict['scan_id'] = scan.id # equal for all nodules/annotations 
            ann_dict['nodule_idx'] = nod_idx 
            ann_dict['annotation_idx'] = ann_idx 
            ann_dict['annotation_num'] = len(nodules) 
            ann_dict['annotation_id'] = ann.id # unique - same annotator has different numbers 
            ann_dict['patient_id'] = scan.patient_id
            ann_dict['study_instance_uid'] = scan.study_instance_uid
            ann_dict['series_instance_uid'] = scan.series_instance_uid
            scan_ann.append(ann_dict)
            
             
            mask_vol = np.zeros(vol.spatial_shape, dtype=np.uint8)
            bbox = ann.bbox()
            mask = ann.boolean_mask()
            
            mask_vol[bbox][mask] = 1 

            ### chatgpt fix
            # z0, z1 = bbox[0]
            # y0, y1 = bbox[1]
            # x0, x1 = bbox[2]
            # subvol = mask_vol[z0:z1, y0:y1, x0:x1]
            # subvol[mask] = 1
            # mask_vol[z0:z1, y0:y1, x0:x1] = subvol
            # apply_mask_to_volume(mask_vol, bbox, mask)

            mask_vol = tio.LabelMap(tensor=mask_vol[None], affine=vol.affine)
            mask_vol.save(path_root_data/path_rel/f'seg_{nod_idx}_{ann_idx}.nii.gz')

        # Perform a consensus consolidation and 50% agreement level.
        cmask,cbbox, masks = consensus(nodules, clevel=0.5)
        mask_vol = np.zeros(vol.spatial_shape, dtype=np.uint8)
        
        mask_vol[cbbox][cmask] = 1 

        ### chatgpt fix
        # z0, z1 = cbbox[0]
        # y0, y1 = cbbox[1]
        # x0, x1 = cbbox[2]
        # subvol = mask_vol[z0:z1, y0:y1, x0:x1]
        # subvol[cmask] = 1
        # mask_vol[z0:z1, y0:y1, x0:x1] = subvol
        # apply_mask_to_volume(mask_vol, cbbox, cmask)

        mask_vol = tio.LabelMap(tensor=mask_vol[None], affine=vol.affine)
        mask_vol.save(path_root_data/path_rel/f'seg_{nod_idx}.nii.gz')

        
    return scan_ann


if __name__ == "__main__":
    scan_ids = range(1, len(list(pl.query(pl.Scan)))+1)
    
    # Settings 
    # path_root = Path('/home/gustav/Coscine_Public/LIDC-IDRI')
    # path_root_download = path_root/'download/TCIA_LIDC-IDRI_20200921/LIDC-IDRI'
    path_root = Path('/radraid2/mwahianwar/LIDC')
    path_root_download = path_root/'data/LIDC-IDRI-pylidc'
    path_root_out = path_root/'preprocessed'
    path_root_data = path_root_out/'data'


    # Option 1: Multi-CPU 
    # all_ann = []
    # with Pool() as pool:
    #     for scan_ann in tqdm(pool.imap_unordered(scan2labels, scan_ids), total=len(scan_ids)):
    #         all_ann.extend(scan_ann)

    # Option 2: Single-CPU (if you need a coffee break)
    all_ann = []
    for scan_id in tqdm(scan_ids):
        all_ann.extend(scan2labels(scan_id))

    df = pd.DataFrame(all_ann)
    df.to_csv(path_root_out/'annotation.csv', index=False)
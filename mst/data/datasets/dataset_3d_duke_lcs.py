from pathlib import Path 
import pandas as pd 
import torch.utils.data as data 
import torchio as tio
import torch  

from .augmentations.augmentations_3d import ImageOrSubjectToTensor, RescaleIntensity, ZNormalization, CropOrPad

class DUKELCS_Dataset3D(data.Dataset):
    # PATH_ROOT = Path('/home/gustav/Coscine_Public/LIDC-IDRI/')
    # PATH_ROOT = Path('/home/gustav/Documents/datasets/LIDC-IDRI/')
    PATH_ROOT = Path('/radraid2/mwahianwar/MST/duke_lcs')
    LABEL = 'Malignant_lbl'

    def __init__(
            self,
            path_root=None,
            fold = 0,
            split= None,
            fraction=None,
            transform = None,
            image_resize = None,
            resample=None,
            flip = False,
            random_rotate=False,
            image_crop = (224, 224, 32),
            random_center=False,
            noise=False, 
            to_tensor = True,
        ):
        self.path_root = self.PATH_ROOT if path_root is None else Path(path_root)
        self.path_root_data = self.path_root/'preprocessed_crop'
        self.split =  split

        # mask_name='mask'
        mask_name=None

        if transform is None: 
            self.transform = tio.Compose([
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.Resample(resample) if resample is not None else tio.Lambda(lambda x: x),
                tio.Lambda(lambda x: x.moveaxis(1, 2)), # Just for viewing, otherwise upside down
                CropOrPad(image_crop, random_center=random_center, mask_name=mask_name, padding_mode='minimum') if image_crop is not None else tio.Lambda(lambda x: x),

                tio.Clamp(-1000, 1000),
                RescaleIntensity((-1,1), in_min_max=(-1000, 1000), per_channel=True),

                # tio.Lambda(lambda x: x.moveaxis(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else tio.Lambda(lambda x: x), # WARNING: 1,2 if Subject, 2, 3 if tensor
                tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0,90), translation=0, isotropic=True, default_pad_value='minimum') if random_rotate else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1,2)) if flip else tio.Lambda(lambda x: x), # WARNING: Padding mask 
                tio.Lambda(lambda x:-x if torch.rand((1,),)[0]<0.5 else x, types_to_apply=[tio.INTENSITY]) if noise else tio.Lambda(lambda x: x),
                tio.RandomNoise(std=(0, 0.1)) if noise else tio.Lambda(lambda x: x),

                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x)             
            ])
        else:
            self.transform = transform


        # Get split file 
        path_csv = self.path_root/'split.csv'
        path_or_stream = path_csv 
        self.df = self.load_split(path_or_stream, fold=fold, split=split, fraction=fraction)#.set_index('scan_id', drop=True)
        self.item_pointers = self.df.index.tolist()

        
    def __len__(self):
        return len(self.item_pointers)
    
    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):   
        return tio.LabelMap(path_img)

    def __getitem__(self, index):
        ### PatientID,SeriesInstanceUID,StudyDate,CoordX,CoordY,CoordZ,LesionID,AnnotationID,NoduleID,label,Age_at_StudyDate,Gender,Malignant,Fold,Split

        uid = self.item_pointers[index]
        item = self.df.loc[uid]
        target =  item[self.LABEL]
        nodule_idx = item['nodule_id']
        # MWW 071625
        # rel_path = Path(item['patient_id'])/item['study_instance_uid']/item['series_instance_uid']
        rel_path = Path(str(item['patient-id']))
        path_dir = self.path_root_data/rel_path

        filename = f'img_{nodule_idx}.nii.gz'
        img_org = self.load_img(path_dir/filename)

        # filename = f'seg_{nodule_idx}.nii.gz'
        # mask = self.load_map(path_dir/filename)
        mask = None
        
        masks = {}
        # if self.split == "test":
        #     for ann_idx in range(item['annotation_num']):
        #         masks[f'mask_{ann_idx}'] = self.load_map(path_dir/f"seg_{nodule_idx}_{ann_idx}.nii.gz" ) 
                    
        
        subj = tio.Subject(img=img_org, mask=mask, **masks)
        subj = self.transform(subj)

        img = subj['img']
        
        if self.split == "test":
            masks = {key: subj[key] for key in masks.keys()}

        return {'uid':uid, 
                'source': img, 
                # 'mask':subj['mask'], 
                # **masks, 
                'target':target, 
                'affine':img_org.affine, 'path':str(rel_path), 'filename':filename}
    

    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer)
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df
import torch
from torch.utils.data import Dataset
import glob
import os
import time
import SimpleITK as sitk
import random
import numpy as np
from .dataset_specifics import *
from monai.transforms.spatial.dictionary import Rand3DElasticd
from collections import defaultdict


class TestDataset(Dataset):

    def __init__(self, args):
        self.image_test = []
        self.dataset = args.dataset
        self.data_root = "/home/prosjekt/PerfusionCT/StrokeSUS/ADNet/"
        self.nii_studies = "nii_studies/" if "DWI" not in self.dataset else "DWI_nii_studies/"
        if args.original_ds: self.nii_studies = "orig_nii_studies/" if "DWI" not in self.dataset else "DWI_nii_studies/"

        # reading the paths
        if args.dataset == 'CMR': self.image_dirs = glob.glob(os.path.join(self.data_root, 'cmr_MR_normalized/image*'))
        elif args.dataset == 'CHAOST2': self.image_dirs = glob.glob(os.path.join(self.data_root, 'chaos_MR_T2_normalized/image*'))
        elif 'CTP' in args.dataset or 'DWI' in args.dataset:
            if args.dataset == "CTP_LVO":
                self.image_dirs = glob.glob(os.path.join(self.data_root, self.nii_studies + 'study_CTP_00*'))+glob.glob(os.path.join(self.data_root, self.nii_studies + 'study_CTP_01*'))
                if not args.original_ds: self.image_dirs += (glob.glob(os.path.join(self.data_root, self.nii_studies + 'study_CTP_20*'))+glob.glob(os.path.join(self.data_root, self.nii_studies + 'study_CTP_21*')))
            elif args.dataset == "CTP_Non-LVO":
                self.image_dirs = glob.glob(os.path.join(self.data_root, self.nii_studies + 'study_CTP_02*'))
                if not args.original_ds: self.image_dirs += glob.glob(os.path.join(self.data_root, self.nii_studies + 'study_CTP_22*'))
            self.image_dirs = glob.glob(os.path.join(self.data_root, self.nii_studies+'study_*'))
            self.test_patients = ["01_001", "01_007", "01_013", "01_019", "01_025", "01_031",
                                  "01_037", "01_044", "01_049", "01_053", "01_061", "01_067", "01_074",
                                  "02_001", "02_007", "02_013", "02_019", "02_025", "02_031", "02_036",
                                  "02_043", "02_050", "02_055", "02_062", "03_003", "03_010", "03_014",
                                  "01_057", "01_059", "01_066", "01_068", "01_071", "01_073"]
            self.exclude_patients = ["21_001", "21_007", "21_013", "21_019", "21_025", "21_031",
                                     "21_037", "21_044", "21_049", "21_053", "21_061", "21_067", "21_074",
                                     "22_001", "22_007", "22_013", "22_019", "22_025", "22_031", "22_036",
                                     "22_043", "22_050", "22_055", "22_062", "23_003", "23_010", "23_014",
                                     "21_057", "21_059", "21_066", "21_068", "21_071", "21_073"]
            self.val_patients = ['21_063', '01_004', '21_016', '21_021', '01_024', '21_029', '21_009', '21_051',
                                 '01_076', '21_018', '01_028', '21_022', '21_050', '21_027', '01_006', '21_004',
                                 '21_002', '21_026', '01_038', '01_033', '21_070', '21_023', '21_028', '20_007',
                                 '01_040', '01_014', '01_034', '21_011', '21_075', '01_029', '01_063', '21_052',
                                 '02_005', '22_015', '02_041', '02_038', '22_040', '02_003', '02_020', '22_016',
                                 '22_060', '22_028', '02_018', '22_045', '22_003', '22_061', '22_042', '22_006',
                                 '02_040', '02_022', '22_024', '02_026', '02_010', '22_004', '03_005', '03_002',
                                 '23_004', '03_004', '23_005', '23_013']

        if 'CTP' not in args.dataset:
            self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
            # remove test fold!
            self.FOLD = get_folds(args.dataset, args.original_ds)
            self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args.fold]]
            self.support_dir = self.image_dirs[-1]
            self.image_dirs = self.image_dirs[:-1]  # remove support
        else:
            self.support_dir = [fold for fold in self.image_dirs if fold.split("study_CTP_")[-1] not in self.val_patients
                                and fold.split("study_CTP_")[-1] not in self.test_patients
                                and fold.split("study_CTP_")[-1] not in self.exclude_patients]
            self.image_dirs = [fold for fold in self.image_dirs if fold.split("study_CTP_")[-1] in self.val_patients]

        if 'CTP' in args.dataset or "DWI" in args.dataset:
            self.image_test = [fold for fold in self.image_dirs if fold.split("study_CTP_")[-1] in self.test_patients]
            # split into support/query
            if args.dataset == "CTP_Non-LVO":
                if args.n_shot==1: self.support_dir = [self.support_dir[2]]
                else: self.support_dir = [self.support_dir[2],self.support_dir[3],self.support_dir[5],self.support_dir[6],self.support_dir[8]]
            else: self.support_dir = self.support_dir[:args.n_shot]  # [self.image_dirs[0]]
            # append the test patients
            for test_p in self.image_test: self.image_dirs.append(test_p)
        # split into support/query
        self.label = None

        # evaluation protocol
        self.EP1 = args.EP1

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        T = 30

        img_path = self.image_dirs[idx]
        if "CTP" not in self.dataset:
            prefix = "image_" if "DWI" not in self.dataset else "study_"
            img = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(img_path)))
            img = (img - img.mean()) / (img.std() + 1e-5)
            img = torch.stack(1 * [img], axis=0)

            lbl = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(img_path.split(prefix)[0] + 'label_' + img_path.split(prefix)[-1])))
            if "DWI" not in self.dataset:
                lbl[lbl == 200] = 1
                lbl[lbl == 500] = 2
                lbl[lbl == 600] = 3
            else: lbl[lbl > 100] = 1
            lbl = 1 * (lbl == self.label)
        else:
            slices = len(glob.glob(img_path + "/*"))
            img = torch.empty((T, slices, 512, 512))
            for slice_idx in range(slices):
                image_path = os.path.join(img_path, str(slice_idx) + ".nii.gz")
                img[:,slice_idx,...] = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))

            # normalize == after that -> mean=0, std=1
            img = (img - img.mean()) / (img.std() + 1e-5)
            lbl_tmp = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(img_path.split('study_')[0] + 'label_' + img_path.split('study_')[-1] + ".nii.gz")))

            div_val = 42
            lbl_tmp[lbl_tmp<=85+div_val] = 0  # background
            lbl_tmp[lbl_tmp>85+div_val] = 1  # hypoperfused region
            # lbl_tmp[np.logical_and(lbl_tmp > 85 + div_val, lbl_tmp <= 170 + div_val)] = 1  # penumbra
            # lbl_tmp[lbl_tmp > 170 + div_val] = 2  # core
            lbl_tmp = 1 * (lbl_tmp == self.label)

            lbl = torch.stack(30 * [lbl_tmp], axis=0)  # stack the ground truth images together
        sample = {'id': img_path}

        if self.EP1:  # Evaluation protocol 1.
            idx = lbl.sum(axis=(1, 2)) > 0
            sample['image'] = img[idx]
            sample['label'] = lbl[idx]
        else:  # Evaluation protocol 2 (default).
            sample['image'] = img
            sample['label'] = lbl

        return sample

    def get_support_index(self, n_shot, C):
        """
        Selecting intervals according to Ouyang et al.
        """
        if n_shot == 1: pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None, all_slices=True, N=None):
        T = 30
        if label is None: raise ValueError('Need to specify label class!')
        if N is None: raise ValueError("Need to specify the number of shots")

        # img_path = self.support_dir
        arr_samples = []
        for img_path in self.support_dir[:N]:  # take the first N images for support
            if "CTP" not in self.dataset:
                prefix = "image_" if "DWI" not in self.dataset else "study_"
                img = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(img_path)))
                img = (img - img.mean()) / (img.std() + 1e-5)
                img = torch.stack(1 * [img], axis=0)

                lbl = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(img_path.split(prefix)[0] + 'label_' + img_path.split(prefix)[-1])))
                if "DWI" not in self.dataset:
                    lbl[lbl == 200] = 1
                    lbl[lbl == 500] = 2
                    lbl[lbl == 600] = 3
                else: lbl[lbl > 100] = 1
                lbl = 1 * (lbl == label)
            else:
                slices = len(glob.glob(img_path + "/*"))
                img = torch.empty((T, slices, 512, 512))
                for slice_idx in range(slices):
                    image_path = os.path.join(img_path, str(slice_idx) + ".nii.gz")
                    img[:,slice_idx,...] = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))
                img = (img - img.mean()) / (img.std() + 1e-5)

                lbl_tmp = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(img_path.split('study_')[0] + 'label_' + img_path.split('study_')[-1] + ".nii.gz")))

                div_val = 42
                lbl_tmp[lbl_tmp <= 85+div_val] = 0  # background
                lbl_tmp[lbl_tmp > 85+div_val] = 1  # hypoperfused region
                # lbl_tmp[np.logical_and(lbl_tmp > 85+div_val, lbl_tmp <= 170+div_val)] = 1 # penumbra
                # lbl_tmp[lbl_tmp > 170+div_val] = 2  # core
                lbl_tmp = 1 * (lbl_tmp == label)
                lbl = torch.stack(30 * [lbl_tmp], axis=0)  # stack the ground truth images together

            sample = {}
            if all_slices:
                sample['image'] = img[None]
                sample['label'] = lbl[None]

                # target = np.where(lbl.sum(axis=(-2, -1)) > 0)[0]
                # mask = np.zeros(lbl.shape) == 1
                # mask[target.astype('float').mean().astype('int')] = True
                # sample['label'] = torch.from_numpy((mask*1)*lbl)[None]
            else:
                # select N labeled slices
                if N is None: raise ValueError('Need to specify number of labeled slices!')
                idx = lbl.sum(axis=(1, 2)) > 0
                idx_ = self.get_support_index(N, idx.sum())
                sample['image'] = img[:, idx][:, idx_][None]
                sample['label'] = lbl[idx][idx_][None]
            arr_samples.append(sample)

        return arr_samples


class TrainDataset(Dataset):

    def __init__(self, args):
        self.n_shot = args.n_shot
        self.n_way = args.n_way
        self.n_query = args.n_query
        self.n_sv = args.n_sv
        self.max_iter = args.max_iterations
        self.min_size = args.min_size
        self.max_slices = args.max_slices
        self.dataset = args.dataset
        self.use_labels_intrain = args.use_labels_intrain
        self.read = True if "CTP" not in self.dataset and "DWI" not in self.dataset else False  # read images before get_item

        self.data_root = "/home/prosjekt/PerfusionCT/StrokeSUS/ADNet/"
        self.nii_studies = "orig_nii_studies/" if "DWI" not in self.dataset else "DWI_nii_studies/"
        self.spv_fold = "supervoxels_ALL"
        self.spv_type = "3D-FELZENSZWALB_PMs_stacked_RGB_v3.0"
        self.spv_mask = "all_MASK"
        self.spv_prefix = 'superpix-3D_felzenszwalb_'
        self.images = {}
        self.sprvxls = {}
        self.valid_spr_slices = {}

        # reading the paths (leaving the reading of images into memory to __getitem__)
        if args.dataset == 'CMR': self.image_dirs = glob.glob(os.path.join(self.data_root, 'cmr_MR_normalized/image*'))
        elif args.dataset == 'CHAOST2': self.image_dirs = glob.glob(os.path.join(self.data_root, 'chaos_MR_T2_normalized/image*'))
        elif 'CTP' in args.dataset or 'DWI' in args.dataset:
            if args.dataset=="CTP_LVO":
                self.image_dirs = glob.glob(os.path.join(self.data_root, self.nii_studies + 'study_CTP_00*'))+glob.glob(os.path.join(self.data_root, self.nii_studies + 'study_CTP_01*'))
                self.sprvxl_dirs = glob.glob(os.path.join(self.data_root, self.nii_studies + 'label_CTP_00*'))+glob.glob(os.path.join(self.data_root, self.nii_studies + 'label_CTP_01*'))
                if not args.original_ds:
                    self.image_dirs += (glob.glob(os.path.join(self.data_root, self.nii_studies + 'study_CTP_20*'))+glob.glob(os.path.join(self.data_root, self.nii_studies + 'study_CTP_21*')))
                    self.sprvxl_dirs += (glob.glob(os.path.join(self.data_root, self.nii_studies + 'label_CTP_20*'))+glob.glob(os.path.join(self.data_root, self.nii_studies + 'label_CTP_21*')))
            elif args.dataset=="CTP_Non-LVO":
                self.image_dirs = glob.glob(os.path.join(self.data_root, self.nii_studies+'study_CTP_02*'))
                self.sprvxl_dirs = glob.glob(os.path.join(self.data_root, self.nii_studies+'label_CTP_02*'))
                if not args.original_ds:
                    self.image_dirs += glob.glob(os.path.join(self.data_root, self.nii_studies + 'study_CTP_22*'))
                    self.sprvxl_dirs += glob.glob(os.path.join(self.data_root, self.nii_studies + 'label_CTP_22*'))
            else:
                self.image_dirs = glob.glob(os.path.join(self.data_root, self.nii_studies+'study_CTP_*'))
                self.sprvxl_dirs = glob.glob(os.path.join(self.data_root, self.nii_studies+'label_CTP_*'))

            self.test_patients = ["01_001", "01_007", "01_013", "01_019", "01_025", "01_031",
                                  "01_037", "01_044", "01_049", "01_053", "01_061", "01_067", "01_074",
                                  "02_001", "02_007", "02_013", "02_019", "02_025", "02_031", "02_036",
                                  "02_043", "02_050", "02_055", "02_062", "03_003", "03_010", "03_014",
                                  "01_057", "01_059", "01_066", "01_068", "01_071", "01_073"]
            self.exclude_patients = ["21_001", "21_007", "21_013", "21_019", "21_025", "21_031",
                                     "21_037", "21_044", "21_049", "21_053", "21_061", "21_067", "21_074",
                                     "22_001", "22_007", "22_013", "22_019", "22_025", "22_031", "22_036",
                                     "22_043", "22_050", "22_055", "22_062", "23_003", "23_010", "23_014",
                                     "21_057", "21_059", "21_066", "21_068", "21_071", "21_073"]
            self.val_patients = ['21_063', '01_004', '21_016', '21_021', '01_024', '21_029', '21_009', '21_051',
                                 '01_076', '21_018', '01_028', '21_022', '21_050', '21_027', '01_006', '21_004',
                                 '21_002', '21_026', '01_038', '01_033', '21_070', '21_023', '21_028', '20_007',
                                 '01_040', '01_014', '01_034', '21_011', '21_075', '01_029', '01_063', '21_052',
                                 '02_005', '22_015', '02_041', '02_038', '22_040', '02_003', '02_020', '22_016',
                                 '22_060', '22_028', '02_018', '22_045', '22_003', '22_061', '22_042', '22_006',
                                 '02_040', '02_022', '22_024', '02_026', '02_010', '22_004', '03_005', '03_002',
                                 '23_004', '03_004', '23_005', '23_013']

        self.FOLD = get_folds(args.dataset, args.original_ds)
        if 'CTP' not in args.dataset:
            self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
            if "DWI" not in args.dataset:
                self.sprvxl_dirs = glob.glob(os.path.join(self.data_root, 'supervoxels_' + str(args.n_sv), 'super*'))
                self.sprvxl_dirs = sorted(self.sprvxl_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
            else:
                self.sprvxl_dirs = glob.glob(os.path.join(self.data_root, self.spv_fold, self.spv_type, self.spv_mask, "CTP_*"))
                idpatients = [fold.split("/study_")[-1].split(".nii.gz")[0] for fold in self.image_dirs]
                self.sprvxl_dirs = [fold for fold in self.sprvxl_dirs if fold.split("/")[-1] in idpatients]
        else:
            self.image_dirs = [fold for fold in self.image_dirs if fold.split("study_")[-1][4:] not in self.test_patients
                               and fold.split("study_")[-1][4:] not in self.exclude_patients]
            if self.use_labels_intrain:
                self.sprvxl_dirs = [fold for fold in self.sprvxl_dirs if fold.split("label_")[-1][4:] not in self.test_patients
                                    and fold.split("label_")[-1][4:] not in self.exclude_patients]
            else:
                self.sprvxl_dirs = glob.glob(os.path.join(self.data_root, self.spv_fold, self.spv_type, self.spv_mask, "CTP_*"))
                idpatients = [fold.split("/study_")[-1].split(".nii.gz")[0] for fold in self.image_dirs]
                self.sprvxl_dirs = [fold for fold in self.sprvxl_dirs if fold.split("/")[-1] in idpatients]

        if "CTP" in args.dataset or "DWI" in args.dataset:
            self.sprvxl_dirs = [fold for fold in self.sprvxl_dirs if fold.split("/")[-1][4:] not in self.test_patients and fold.split("/")[-1][4:] not in self.exclude_patients]
            self.image_dirs.sort()
            self.sprvxl_dirs.sort()

            val_idx = [idx for idx, fold in enumerate(self.image_dirs) if fold.split("study_")[-1][4:] in self.val_patients]
            self.FOLD[args.fold] = set(val_idx)  # the validation index becomes the FOLD and are excluded from

        # remove test fold!
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx not in self.FOLD[args.fold]]
        self.sprvxl_dirs = [elem for idx, elem in enumerate(self.sprvxl_dirs) if idx not in self.FOLD[args.fold]]
        self.N = len(self.image_dirs)

        # read images
        if self.read:
            for image_dir, sprvxl_dir in zip(self.image_dirs, self.sprvxl_dirs):
                img = sitk.ReadImage(image_dir)
                self.res = img.GetSpacing()
                img = sitk.GetArrayFromImage(img)
                self.images[image_dir] = torch.from_numpy(img)
                spr = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_dir)))
                self.sprvxls[sprvxl_dir] = spr

                unique = list(torch.unique(spr))
                unique.remove(0)
                self.valid_spr_slices[image_dir] = []
                for val in unique:
                    spr_val = (spr == val)

                    n_slices = min(spr_val.shape[0], self.max_slices)
                    sample_list = []
                    for r in range(spr_val.shape[0] - (n_slices - 1)):
                        sample_idx = torch.arange(r, r + n_slices).tolist()
                        candidate = spr_val[sample_idx]
                        if candidate.sum() > self.min_size: sample_list.append(sample_idx)
                    if len(sample_list) > 0: self.valid_spr_slices[image_dir].append((val, sample_list))

        # set transformation details
        rad = 5 * (np.pi / 180)
        self.rand_3d_elastic = Rand3DElasticd(
            keys=("img", "seg"),
            mode=("bilinear", "nearest"),
            sigma_range=(5, 5),
            magnitude_range=(0, 0),
            prob=1.0,  # because probability controlled by this class
            rotate_range=(rad, rad, rad),
            shear_range=(rad, rad, rad),
            translate_range=(5, 5, 1),
            scale_range=((-0.1, 0.2), (-0.1, 0.2), (-0.1, 0.2)),
            as_tensor_output=True,
            device='cpu')

    def __len__(self):
        return self.max_iter

    def gamma_tansform(self, img):
        gamma_range = (0.5, 1.5)
        gamma = torch.rand(1) * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = (img.max() - cmin + 1e-5)

        img = img - cmin + 1e-5
        img = irange * torch.pow(img * 1.0 / irange, gamma)
        img = img + cmin

        return img

    def __getitem__(self, idx):
        T = 30
        spr = None
        # sample patient idx
        pat_idx = random.choice(range(len(self.image_dirs)))

        if self.read:  # get image/supervoxel volume from dictionary
            img = self.images[self.image_dirs[pat_idx]]
            sprvxl = self.sprvxls[self.sprvxl_dirs[pat_idx]]
        else:  # read image/supervoxel volume into memory
            assert "CTP" in self.dataset, "Dataset is not CTP"
            # select the vol images and the slice idx
            slices = len(glob.glob(self.image_dirs[pat_idx] + "/*"))
            slice_selected = random.choice(range(slices))
            img = torch.empty((T, slices, 512, 512))
            # set the supervoxel with the corresponding images
            sprvxl = torch.empty((T, slices, 512, 512))
            unique = []
            for slice_idx in range(slices):
                # get the correct slice index
                slc = str(slice_idx + 1)
                if len(slc) == 1: slc = "0" + slc
                # get the supervoxel vol and extract the unique values
                image_path = os.path.join(self.image_dirs[pat_idx], str(slice_idx) + ".nii.gz")
                img[:, slice_idx, ...] = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))
                sprvxl_path = os.path.join(self.sprvxl_dirs[pat_idx], self.spv_prefix + str(self.n_sv) + "_" + slc + ".nii.gz")
                spr = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_path))[:, :, 0])
                sprvxl[:, slice_idx, ...] = torch.stack(T * [spr], axis=0)
                unique.extend(list(torch.unique(spr)))
                unique.remove(0)
            # get the unique values from the list of unique values from the supervoxels
            unique = list(set(unique))
            self.valid_spr_slices[self.image_dirs[pat_idx]] = []

            # for val in unique:
            while len(self.valid_spr_slices[self.image_dirs[pat_idx]])==0:
                random.shuffle(unique)
                for idx_val in range(int(len(unique)/20)):
                    val = unique[idx_val]
                    spr_val = (sprvxl == val)

                    n_slices = min(spr_val.shape[1], self.max_slices)
                    sample_list = []
                    for r in range(spr_val.shape[1] - (n_slices - 1)):
                        sample_idx = torch.arange(r, r+n_slices).tolist()
                        candidate = spr_val[sample_idx]
                        if candidate.sum() > self.min_size*T: sample_list.append(sample_idx)
                    if len(sample_list) > 0: self.valid_spr_slices[self.image_dirs[pat_idx]].append((val, sample_list))

        # normalize == after that -> mean=0, std=1
        img = (img - img.mean()) / (img.std() + 1e-5)
        if self.read: img = torch.from_numpy(img)
        # sample supervoxel
        valid = self.valid_spr_slices[self.image_dirs[pat_idx]]
        cls_idx, candidates = valid[random.randint(0, len(valid) - 1)]

        sprvxl = 1 * (sprvxl == cls_idx)

        sup_lbl = torch.clone(sprvxl)
        qry_lbl = torch.clone(sprvxl)

        sup_img = torch.clone(img)
        qry_img = torch.clone(img)

        # gamma transform
        if np.random.random(1) > 0.5: qry_img = self.gamma_tansform(qry_img)
        else: sup_img = self.gamma_tansform(sup_img)

        # geom transform
        if np.random.random(1) > 0.5:
            for t in range(T):
                res = self.rand_3d_elastic({"img": qry_img[t,...].permute(1, 2, 0), "seg": qry_lbl[t,...].permute(1, 2, 0)})

                qry_img[t,...] = res["img"].permute(2, 0, 1)
                qry_lbl[t,...] = res["seg"].permute(2, 0, 1)

                if t==0:  # only the first time we set the idx
                    # support not tformed
                    constant_s = random.randint(0, len(candidates) - 1)
                    idx_s = candidates[constant_s]

                    k = 50
                    constant_q = constant_s + random.randint(-min(constant_s, k), min(len(candidates) - constant_s - 1, k))
                    idx_q = candidates[constant_q]

        else:
            update_ = False
            for t in range(T):
                if t > 0 and not update_: break
                res = self.rand_3d_elastic({"img": sup_img[t,...].permute(1, 2, 0), "seg": sup_lbl[t,...].permute(1, 2, 0)})
                if t>0 and update_:
                    sup_img[t,...] = res["img"].permute(2, 0, 1)
                    sup_lbl[t,...] = res["seg"].permute(2, 0, 1)

                if t==0:
                    sup_img_ = res["img"].permute(2, 0, 1)
                    sup_lbl_ = res["seg"].permute(2, 0, 1)
                    constant_q = random.randint(0, len(candidates) - 1)
                    idx_q = candidates[constant_q]

                    k = 50
                    constant_s = constant_q + random.randint(-min(constant_q, k), min(len(candidates) - constant_q - 1, k))
                    idx_s = candidates[constant_s]
                    if sup_lbl_[idx_s].sum() > 0:
                        sup_img[t,...] = sup_img_
                        sup_lbl[t,...] = sup_lbl_
                        update_ = True

        sup_lbl = sup_lbl[0,idx_s,...]
        qry_lbl = qry_lbl[0,idx_q,...]

        sup_img = sup_img[:,idx_s,...]
        qry_img = qry_img[:,idx_q,...]

        # b = 215
        # k = 0
        # horizontal_s, vertical_s = sample_xy(sup_lbl, k=k, b=b)
        # horizontal_q, vertical_q = sample_xy(qry_lbl, k=k, b=b)
        #
        # sup_img = sup_img[:, horizontal_s:horizontal_s + b, vertical_s:vertical_s + b]
        # sup_lbl = sup_lbl[:, horizontal_s:horizontal_s + b, vertical_s:vertical_s + b]
        # qry_img = qry_img[:, horizontal_q:horizontal_q + b, vertical_q:vertical_q + b]
        # qry_lbl = qry_lbl[:, horizontal_q:horizontal_q + b, vertical_q:vertical_q + b]

        sample = {'support_images': torch.stack(1 * [sup_img], dim=0),
                  'support_fg_labels': sup_lbl[None],
                  'query_images': torch.stack(1 * [qry_img], dim=0),
                  'query_labels': qry_lbl}

        return sample

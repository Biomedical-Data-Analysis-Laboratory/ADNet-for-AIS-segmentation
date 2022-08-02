import torch
from torch.utils.data import Dataset
import torchvision.transforms as deftfx
import glob
import os
import SimpleITK as sitk
import random
import numpy as np
from . import image_transforms as myit
from .dataset_specifics import *


class TestDataset(Dataset):

    def __init__(self, args):
        self.image_test = []
        self.dataset = args.dataset
        self.nii_studies = "nii_studies/" if "DWI" not in self.dataset else "DWI_nii_studies/"
        if args.original_ds: self.nii_studies = "orig_nii_studies/" if "DWI" not in self.dataset else "DWI_nii_studies/"

        # reading the paths
        if args.dataset == 'CMR':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'cmr_MR_normalized/image*'))
        elif args.dataset == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'chaos_MR_T2_normalized/image*'))
        elif 'CTP' in args.dataset or 'DWI' in args.dataset:
            if args.dataset == "CTP_LVO":
                self.image_dirs = glob.glob(os.path.join(args.data_root, self.nii_studies + 'study_CTP_00*'))+glob.glob(os.path.join(args.data_root, self.nii_studies + 'study_CTP_01*'))
                if not args.original_ds: self.image_dirs += (glob.glob(os.path.join(args.data_root, self.nii_studies + 'study_CTP_20*'))+glob.glob(os.path.join(args.data_root, self.nii_studies + 'study_CTP_21*')))
            elif args.dataset == "CTP_Non-LVO":
                self.image_dirs = glob.glob(os.path.join(args.data_root, self.nii_studies + 'study_CTP_02*'))
                if not args.original_ds: self.image_dirs += glob.glob(os.path.join(args.data_root, self.nii_studies + 'study_CTP_22*'))
            self.image_dirs = glob.glob(os.path.join(args.data_root, self.nii_studies+'study_*'))
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
        else:
            self.support_dir = [fold for fold in self.image_dirs if fold.split("study_CTP_")[-1] not in self.val_patients
                                and fold.split("study_CTP_")[-1] not in self.test_patients
                                and fold.split("study_CTP_")[-1] not in self.exclude_patients]
            self.image_dirs = [fold for fold in self.image_dirs if fold.split("study_CTP_")[-1] in self.val_patients]

        if 'CTP' in args.dataset or "DWI" in args.dataset:
            self.image_test = [fold for fold in self.image_dirs if fold.split("study_CTP_")[-1] in self.test_patients]

        # remove test fold!
        if 'CTP' not in args.dataset:
            self.FOLD = get_folds(args.dataset, args.original_ds)
            self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args.fold]]

        # split into support/query
        if args.dataset == "CTP_Non-LVO":
            if args.n_shot==1: self.support_dir = [self.support_dir[2]]
            else: self.support_dir = [self.support_dir[2],self.support_dir[3],self.support_dir[5],self.support_dir[6],self.support_dir[8]]
        else: self.support_dir = self.support_dir[0:args.n_shot]  # [self.image_dirs[0]]
        # self.image_dirs = self.image_dirs[6:]  # remove support
        # append the test patients
        for test_p in self.image_test: self.image_dirs.append(test_p)
        self.label = None

        # evaluation protocol
        self.EP1 = args.EP1

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):

        img_path = self.image_dirs[idx]
        if "CTP" not in self.dataset:
            prefix = "image_" if "DWI" not in self.dataset else "study_"
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            img = (img - img.mean()) / (img.std() + 1e-5)
            img = np.stack(3 * [img], axis=1)

            lbl = sitk.GetArrayFromImage(sitk.ReadImage(img_path.split(prefix)[0] + 'label_' + img_path.split(prefix)[-1]))
            if "DWI" not in self.dataset:
                lbl[lbl == 200] = 1
                lbl[lbl == 500] = 2
                lbl[lbl == 600] = 3
            else: lbl[lbl > 100] = 1
            lbl = 1 * (lbl == self.label)
        else:
            slices = len(glob.glob(img_path + "/*"))
            img = np.empty((slices, 30, 512, 512))
            for slice_idx in range(slices):
                image_path = os.path.join(img_path, str(slice_idx) + ".nii.gz")
                img[slice_idx,...] = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
            img = (img - img.mean()) / (img.std() + 1e-5)

            lbl_tmp = sitk.GetArrayFromImage(
                sitk.ReadImage(img_path.split('study_')[0] + 'label_' + img_path.split('study_')[-1] + ".nii.gz"))

            div_val = 42
            lbl_tmp[lbl_tmp <= 85+div_val] = 0  # background
            lbl_tmp[lbl_tmp > 85+div_val] = 1  # hypoperfused region
            # lbl_tmp[np.logical_and(lbl_tmp > 85 + div_val, lbl_tmp <= 170 + div_val)] = 1  # penumbra
            # lbl_tmp[lbl_tmp > 170 + div_val] = 2  # core
            lbl_tmp = 1 * (lbl_tmp == self.label)

            lbl = np.stack(30 * [lbl_tmp], axis=1)  # stack the ground truth images together

        sample = {'id': img_path}

        # Evaluation protocol 1.
        if self.EP1:
            idx = lbl.sum(axis=(1, 2)) > 0
            sample['image'] = torch.from_numpy(img[idx])
            sample['label'] = torch.from_numpy(lbl[idx])

        # Evaluation protocol 2 (default).
        else:
            sample['image'] = torch.from_numpy(img)
            sample['label'] = torch.from_numpy(lbl)

        return sample

    def get_support_index(self, n_shot, C):
        """
        Selecting intervals according to Ouyang et al.
        """
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None: raise ValueError('Need to specify label class!')

        arr_samples = []
        if N is None: raise ValueError("Need to specify the number of shots")

        for img_path in self.support_dir[:N]: # take the first N images for support

            # img_path = self.support_dir
            if "CTP" not in self.dataset:
                prefix = "image_" if "DWI" not in self.dataset else "study_"
                img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                img = (img - img.mean()) / (img.std() + 1e-5)
                img = np.stack(3 * [img], axis=1)

                lbl = sitk.GetArrayFromImage(sitk.ReadImage(img_path.split(prefix)[0] + 'label_' + img_path.split(prefix)[-1]))
                if "DWI" not in self.dataset:
                    lbl[lbl == 200] = 1
                    lbl[lbl == 500] = 2
                    lbl[lbl == 600] = 3
                else: lbl[lbl > 100] = 1
                lbl = 1 * (lbl == label)
            else:
                slices = len(glob.glob(img_path + "/*"))
                img = np.empty((slices, 30, 512, 512))
                for slice_idx in range(slices):
                    image_path = os.path.join(img_path, str(slice_idx) + ".nii.gz")
                    img[slice_idx,...] = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
                img = (img - img.mean()) / (img.std() + 1e-5)

                lbl_tmp = sitk.GetArrayFromImage(
                    sitk.ReadImage(img_path.split('study_')[0] + 'label_' + img_path.split('study_')[-1] + ".nii.gz"))

                div_val = 42
                lbl_tmp[lbl_tmp <= 85+div_val] = 0  # background
                lbl_tmp[lbl_tmp > 85+div_val] = 1  # hypoperfused region
                # lbl_tmp[np.logical_and(lbl_tmp > 85+div_val, lbl_tmp <= 170+div_val)] = 1 # penumbra
                # lbl_tmp[lbl_tmp > 170+div_val] = 2  # core
                lbl_tmp = 1 * (lbl_tmp == label)
                lbl = np.stack(30 * [lbl_tmp], axis=1)  # stack the ground truth images together

            sample = {}
            if all_slices:
                sample['image'] = torch.from_numpy(img)
                sample['label'] = torch.from_numpy(lbl)
            else:
                # select N labeled slices
                if N is None: raise ValueError('Need to specify number of labeled slices!')
                idx = lbl.sum(axis=(-2, -1)) > 0
                idx_ = self.get_support_index(N, idx.sum())
                sample['image'] = torch.from_numpy(img[idx][idx_])
                sample['label'] = torch.from_numpy(lbl[idx][idx_])
            arr_samples.append(sample)

        return arr_samples


class TrainDataset(Dataset):

    def __init__(self, args):
        self.n_shot = args.n_shot
        self.n_way = args.n_way
        self.n_query = args.n_query
        self.n_sv = args.n_sv
        self.max_iter = args.max_iterations
        self.dataset = args.dataset
        self.read = True if "CTP" not in self.dataset and "DWI" not in self.dataset else False  # read images before get_item
        self.train_sampling = 'neighbors'
        self.min_size = args.min_size
        self.seed = args.seed
        self.use_labels_intrain = args.use_labels_intrain
        self.nii_studies = "orig_nii_studies/" if "DWI" not in self.dataset else "DWI_nii_studies/"
        self.spv_fold = "supervoxels_ALL"
        self.spv_type = "3D-FELZENSZWALB_PMs_stacked_RGB_v3.0"
        self.spv_mask = "all_MASK"
        self.spv_prefix = 'superpix-3D_felzenszwalb_'

        # reading the paths (leaving the reading of images into memory to __getitem__)
        if args.dataset == 'CMR': self.image_dirs = glob.glob(os.path.join(args.data_root, 'cmr_MR_normalized/image*'))
        elif args.dataset == 'CHAOST2': self.image_dirs = glob.glob(os.path.join(args.data_root, 'chaos_MR_T2_normalized/image*'))
        elif 'CTP' in args.dataset or 'DWI' in args.dataset:
            if args.dataset=="CTP_LVO":
                self.image_dirs = glob.glob(os.path.join(args.data_root, self.nii_studies + 'study_CTP_00*'))+glob.glob(os.path.join(args.data_root, self.nii_studies + 'study_CTP_01*'))
                self.sprvxl_dirs = glob.glob(os.path.join(args.data_root, self.nii_studies + 'label_CTP_00*'))+glob.glob(os.path.join(args.data_root, self.nii_studies + 'label_CTP_01*'))
                if not args.original_ds:
                    self.image_dirs += (glob.glob(os.path.join(args.data_root, self.nii_studies + 'study_CTP_20*'))+glob.glob(os.path.join(args.data_root, self.nii_studies + 'study_CTP_21*')))
                    self.sprvxl_dirs += (glob.glob(os.path.join(args.data_root, self.nii_studies + 'label_CTP_20*'))+glob.glob(os.path.join(args.data_root, self.nii_studies + 'label_CTP_21*')))
            elif args.dataset=="CTP_Non-LVO":
                self.image_dirs = glob.glob(os.path.join(args.data_root, self.nii_studies+'study_CTP_02*'))
                self.sprvxl_dirs = glob.glob(os.path.join(args.data_root, self.nii_studies+'label_CTP_02*'))
                if not args.original_ds:
                    self.image_dirs += glob.glob(os.path.join(args.data_root, self.nii_studies + 'study_CTP_22*'))
                    self.sprvxl_dirs += glob.glob(os.path.join(args.data_root, self.nii_studies + 'label_CTP_22*'))
            else:
                self.image_dirs = glob.glob(os.path.join(args.data_root, self.nii_studies+'study_CTP_*'))
                self.sprvxl_dirs = glob.glob(os.path.join(args.data_root, self.nii_studies+'label_CTP_*'))

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
                self.sprvxl_dirs = glob.glob(os.path.join(args.data_root, 'supervoxels_' + str(args.n_sv), 'super*'))
                self.sprvxl_dirs = sorted(self.sprvxl_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
            else:
                self.sprvxl_dirs = glob.glob(os.path.join(args.data_root, self.spv_fold, self.spv_type, self.spv_mask, "CTP_*"))
                idpatients = [fold.split("/study_")[-1].split(".nii.gz")[0] for fold in self.image_dirs]
                self.sprvxl_dirs = [fold for fold in self.sprvxl_dirs if fold.split("/")[-1] in idpatients]
        else:
            self.image_dirs = [fold for fold in self.image_dirs if fold.split("study_")[-1][4:] not in self.test_patients
                               and fold.split("study_")[-1][4:] not in self.exclude_patients]
            if self.use_labels_intrain:
                self.sprvxl_dirs = [fold for fold in self.sprvxl_dirs if fold.split("label_")[-1][4:] not in self.test_patients
                                    and fold.split("label_")[-1][4:] not in self.exclude_patients]
            else:
                self.sprvxl_dirs = glob.glob(os.path.join(args.data_root, self.spv_fold, self.spv_type, self.spv_mask, "CTP_*"))
                idpatients = [fold.split("/study_")[-1].split(".nii.gz")[0] for fold in self.image_dirs]
                self.sprvxl_dirs = [fold for fold in self.sprvxl_dirs if fold.split("/")[-1] in idpatients]

        if "CTP" in args.dataset or "DWI" in args.dataset:
            self.sprvxl_dirs = [fold for fold in self.sprvxl_dirs if fold.split("/")[-1][4:] not in self.test_patients and fold.split("/")[-1][4:] not in self.exclude_patients]
            self.image_dirs.sort()
            self.sprvxl_dirs.sort()

            val_idx = [idx for idx, fold in enumerate(self.image_dirs) if fold.split("study_")[-1][4:] in self.val_patients]
            self.FOLD[args.fold] = set(val_idx)  # the validation index becomes the FOLD and are excluded from

        # remove val/test fold!
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx not in self.FOLD[args.fold]]
        self.sprvxl_dirs = [elem for idx, elem in enumerate(self.sprvxl_dirs) if idx not in self.FOLD[args.fold]]

        # read images
        if self.read:
            self.images = {}
            self.sprvxls = {}
            for image_dir, sprvxl_dir in zip(self.image_dirs, self.sprvxl_dirs):
                self.images[image_dir] = sitk.GetArrayFromImage(sitk.ReadImage(image_dir))
                self.sprvxls[sprvxl_dir] = sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_dir))

    def __len__(self):
        return self.max_iter

    def gamma_tansform(self, img):
        gamma_range = (0.5, 1.5)
        gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = (img.max() - cmin + 1e-5)

        img = img - cmin + 1e-5
        img = irange * np.power(img * 1.0 / irange, gamma)
        img = img + cmin

        return img

    def geom_transform(self, img, mask, sprvxl=None):

        affine = {'rotate': 5, 'shift': (5, 5), 'shear': 5, 'scale': (0.9, 1.2)}
        alpha = 10
        sigma = 5
        order = 3

        tfx = [myit.RandomFlip3D(h=True, v=True, t=False, p=0.5),
               myit.RandomAffine(affine.get('rotate'), affine.get('shift'), affine.get('shear'), affine.get('scale'), affine.get('scale_iso', True), order=order),
               myit.ElasticTransform(alpha, sigma)]
        transform = deftfx.Compose(tfx)

        n_chann = 3 if "CTP" not in self.dataset else 30
        if len(img.shape) > 4:  # support images
            n_shot = img.shape[1]
            for shot in range(n_shot):
                mask_forcat = mask[0, shot] if "CTP" in self.dataset else mask[:, shot]
                cat = np.concatenate((img[0, shot], mask_forcat)).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[0, shot] = cat[:n_chann, :, :]
                mask[:, shot] = np.rint(cat[n_chann:, :, :])
        else:  # query images
            for q in range(img.shape[0]):
                mask_forcat = mask[q] if "CTP" in self.dataset else mask[q][None]
                cat = np.concatenate((img[q], mask_forcat, sprvxl)).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[q] = cat[:n_chann, :, :]
                mask[q] = np.rint(cat[n_chann:n_chann*2, :, :].squeeze()) if "CTP" in self.dataset else np.rint(cat[n_chann:n_chann+1, :, :].squeeze())
                sprvxl = cat[n_chann*2:,:,:] if "CTP" in self.dataset else cat[n_chann+1:,:,:]
        return img, mask, sprvxl

    def __getitem__(self, idx):

        # sample patient idx
        pat_idx = random.choice(range(len(self.image_dirs)))
        slice_selected = -1
        if self.read:
            # get image/supervoxel volume from dictionary
            img = self.images[self.image_dirs[pat_idx]]
            sprvxl = self.sprvxls[self.sprvxl_dirs[pat_idx]]
        else:
            # read image/supervoxel volume into memory
            if "CTP" not in self.dataset:
                img = sitk.GetArrayFromImage(sitk.ReadImage(self.image_dirs[pat_idx]))
                if "DWI" not in self.dataset: sprvxl = sitk.GetArrayFromImage(sitk.ReadImage(self.sprvxl_dirs[pat_idx]))
                else:
                    sprvxl = np.empty(img.shape)
                    for slice_idx in range(img.shape[0]):
                        slc = str(slice_idx + 1)
                        if len(slc) == 1: slc = "0" + slc
                        sprvxl_path = os.path.join(self.sprvxl_dirs[pat_idx], self.spv_prefix + str(self.n_sv) + "_" + slc + ".nii.gz")
                        sprvxl[slice_idx, ...] = sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_path))[:, :, 0]
            else:
                # sample slice idx
                slices = len(glob.glob(self.image_dirs[pat_idx] + "/*"))
                slice_selected = random.choice(range(slices))
                image_path = os.path.join(self.image_dirs[pat_idx], str(slice_selected) + ".nii.gz")
                img = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

                if self.use_labels_intrain:
                    sprvxl = sitk.GetArrayFromImage(sitk.ReadImage(self.sprvxl_dirs[pat_idx]))
                else:
                    sprvxl = np.empty((slices, img.shape[1], img.shape[2]))

                    for slice_idx in range(slices):
                        slc = str(slice_idx + 1)
                        if len(slc) == 1: slc = "0" + slc

                        # sprvxl_path = os.path.join(self.sprvxl_dirs[pat_idx],'superpix-MIDDLE_'+slc+".nii.gz")
                        sprvxl_path = os.path.join(self.sprvxl_dirs[pat_idx], self.spv_prefix + str(self.n_sv) + "_" + slc + ".nii.gz")
                        sprvxl[slice_idx, ...] = sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_path))[:, :, 0]

        # normalize == after that -> mean=0, std=1
        img = (img - img.mean()) / (img.std() + 1e-5)

        # sample class(es) (supervoxel)
        unique = list(np.unique(sprvxl))
        unique.remove(0)
        if self.use_labels_intrain: unique.remove(85)  # remove the brain

        size, cls_idx = 0, -1
        while size < self.min_size:
            n_slices = (self.n_shot * self.n_way) + self.n_query - 1
            while n_slices < ((self.n_shot * self.n_way) + self.n_query):
                if len(unique)==0: return {}
                cls_idx = random.choice(unique)
                unique.remove(cls_idx)

                # extract slices containing the sampled class
                sli_idx = np.sum(sprvxl == cls_idx, axis=(1, 2)) > 0
                # if the selected slice for the support is NOT included in the extract slices don't update n_slices
                if "CTP" in self.dataset and not sli_idx[slice_selected]: continue
                n_slices = np.sum(sli_idx)

            img_slices = img[sli_idx] if "CTP" not in self.dataset else img
            sprvxl_slices = 1 * (sprvxl[sli_idx] == cls_idx)
            # update the slice_selected index subtracting the slices that don't include the selected class before the slice_selected
            slice_selected = slice_selected - (len(sli_idx[:slice_selected]) - sum(sli_idx[:slice_selected]))
            sprvxl_toexp = sprvxl[sli_idx]

            # sample support and query slices
            if "CTP" not in self.dataset:
                i = random.choice(np.arange(n_slices - ((self.n_shot * self.n_way) + self.n_query) + 1)) # successive slices
                sample = np.arange(i, i + (self.n_shot * self.n_way) + self.n_query)
            else: sample = np.array([slice_selected,slice_selected])  # if CTP then take the selected slice

            assert len(sample)==2,"Len of sample is != 2"

            img_to_check = sprvxl_slices[sample[0]]+sprvxl_slices[sample[1]] if "CTP" not in self.dataset else \
                img[0]*(sprvxl_slices[sample[0]] == 1)*1 + img[0]*(sprvxl_slices[sample[1]] == 1)*1
            size = np.sum(img_to_check) / 2

        # invert order
        if np.random.random(1) > 0.5: sample = sample[::-1]  # successive slices (inverted)

        sup_lbl = sprvxl_slices[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
        qry_lbl = sprvxl_slices[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W
        sprvxl_toexp = sprvxl_toexp[sample[self.n_shot * self.n_way:]]
        if "CTP" not in self.dataset:
            sup_img = img_slices[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
            sup_img = np.stack((sup_img, sup_img, sup_img), axis=2)
            qry_img = img_slices[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W
            qry_img = np.stack((qry_img, qry_img, qry_img), axis=1)
        else:
            sup_lbl = np.stack((sup_lbl,) * img_slices.shape[0], axis=2)
            qry_lbl = np.stack((qry_lbl,) * img_slices.shape[0], axis=1)
            sup_img = img_slices[None, None,]  # take the entire timepoints and add additional dimensions
            qry_img = img_slices[None,]

        # print("support_img", sup_img.min(), sup_img.max(), sup_img.shape, sprvxl_toexp.shape)
        # print("query_img", qry_img.min(), qry_img.max(), qry_img.shape)

        # gamma transform
        if np.random.random(1) > 0.5: qry_img = self.gamma_tansform(qry_img)
        else: sup_img = self.gamma_tansform(sup_img)

        # print("--------AFTER GAMMA TRANSF.--------")
        # print("support_img", sup_img.min(), sup_img.max(), sup_img.shape)
        # print("query_img", qry_img.min(), qry_img.max(), qry_img.shape)

        # geom transform
        if np.random.random(1) > 0.5: qry_img, qry_lbl, sprvxl_toexp = self.geom_transform(qry_img, qry_lbl, sprvxl_toexp)
        else: sup_img, sup_lbl, sprvxl_toexp = self.geom_transform(sup_img, sup_lbl, sprvxl_toexp)

        # print("--------AFTER GEOMETRIC TRANSF.--------")
        # print("support_img", sup_img.min(), sup_img.max(), sup_img.shape)
        # print("query_img", qry_img.min(), qry_img.max(), qry_img.shape)

        sample = {'support_images': sup_img,  # (1,1,30,512,512)
                  'support_fg_labels': sup_lbl,  #
                  'query_images': qry_img,  # (1,1,30,512,512)
                  'query_labels': qry_lbl,
                  'sprvxl_toexp': sprvxl_toexp}

        return sample

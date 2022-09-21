import torch
import random


def get_label_names(dataset):
    label_names = {}
    if dataset == 'CMR':
        label_names[0] = 'BG'
        label_names[1] = 'LV-MYO'
        label_names[2] = 'LV-BP'
        label_names[3] = 'RV'
    elif dataset == 'CHAOST2':
        label_names[0] = 'BG'
        label_names[1] = 'LIVER'
        label_names[2] = 'RK'
        label_names[3] = 'LK'
        label_names[4] = 'SPLEEN'
    elif "CTP" in dataset:
        label_names[0] = "BG"
        label_names[1] = "hypoperfused"
        # label_names[1] = "penumbra"
        # label_names[2] = "core"
    elif "DWI" in dataset:
        label_names[0] = "BG"
        label_names[1] = "core"
    return label_names


def get_folds(dataset, original_ds):
    FOLD = {}
    if dataset == 'CMR':
        FOLD[0] = set(range(0, 8))
        FOLD[1] = set(range(7, 15))
        FOLD[2] = set(range(14, 22))
        FOLD[3] = set(range(21, 29))
        FOLD[4] = set(range(28, 35))
        FOLD[4].update([0])
        return FOLD
    elif dataset == 'CHAOST2':
        FOLD[0] = set(range(0, 5))
        FOLD[1] = set(range(4, 9))
        FOLD[2] = set(range(8, 13))
        FOLD[3] = set(range(12, 17))
        FOLD[4] = set(range(16, 20))
        FOLD[4].update([0])
        return FOLD
    elif dataset == "CTP":
        if original_ds: FOLD[0] = set(range(0,152))  # train
        else: FOLD[0] = set(range(0,304))  # train
        # FOLD[0] = set(range(0, 62))
        # FOLD[1] = set(range(61, 123))
        # FOLD[2] = set(range(122, 184))
        # FOLD[3] = set(range(183, 245))
        # FOLD[4] = set(range(244, 304))
        # FOLD[4].update([0])
        return FOLD
    elif dataset == "CTP_LVO":
        if original_ds: FOLD[0] = set(range(0, 77))  # train
        else: FOLD[0] = set(range(0, 77)).union(range(152,229))  # train
        return FOLD
    elif dataset == "CTP_Non-LVO":
        if original_ds: FOLD[0] = set(range(77,137))  # train
        else: FOLD[0] = set(range(77,137)).union(range(229,289))  # train
        return FOLD
    elif dataset == "DWI":
        if original_ds: FOLD[0] = set(range(0,110))
        else: FOLD[0] = set(range(0,220))
        return FOLD
    elif dataset == "DWI_LVO":
        if original_ds: FOLD[0] = set(range(0,64))
        else: FOLD[0] = set(range(0,220))
        return FOLD
    elif dataset == "DWI_Non-LVO":
        if original_ds: FOLD[0] = set(range(64,110))
        else: FOLD[0] = set(range(0,220))
        return FOLD
    else:
        raise ValueError(f'Dataset: {dataset} not found')


def sample_xy(spr, k=0, b=215):

    _, h, v = torch.where(spr)

    if len(h) == 0 or len(v) == 0:
        horizontal = 0
        vertical = 0
    else:

        h_min = min(h)
        h_max = max(h)
        if b > (h_max - h_min):
            kk = min(k, int((h_max - h_min) / 2))
            horizontal = random.randint(max(h_max - b - kk, 0), min(h_min + kk, spr.shape[-1] - b - 1))
        else:
            kk = min(k, int(b / 2))
            horizontal = random.randint(max(h_min - kk, 0), min(h_max - b + kk, spr.shape[-1] - b - 1))

        v_min = min(v)
        v_max = max(v)
        if b > (v_max - v_min):
            kk = min(k, int((v_max - v_min) / 2))
            vertical = random.randint(max(v_max - b - kk, 0), min(v_min + kk, spr.shape[-1] - b - 1))
        else:
            kk = min(k, int(b / 2))
            vertical = random.randint(max(v_min - kk, 0), min(v_max - b + kk, spr.shape[-1] - b - 1))

    return horizontal, vertical

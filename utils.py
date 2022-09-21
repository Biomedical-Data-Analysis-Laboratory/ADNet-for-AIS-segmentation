import torch
import logging
import os
import wandb
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def init_wandb(args):
    wandb.init(project="ADNet", entity="lucatomasetti")

    # Save run name.
    # wandb.run.save()
    # run_name = wandb.run.name

    # Log args.
    config = wandb.config
    config.update(args)

    # return run_name


def set_logger(log_path, file_name):
    if not os.path.isdir(log_path): os.makedirs(log_path)
    path = os.path.join(log_path, file_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Log to .txt
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger


def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    minval = arr.min()
    maxval = arr.max()
    if minval != maxval:
        arr -= minval
        arr *= (255.0 / (maxval - minval))
    return arr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def IOU(preds, label):
    tp = torch.sum((label == 1) * (preds == 1), dtype=torch.float32)
    tn = torch.sum((label == 0) * (preds == 0), dtype=torch.float32)
    fp = torch.sum((label == 0) * (preds == 1), dtype=torch.float32)
    fn = torch.sum((label == 1) * (preds == 0), dtype=torch.float32)
    iou = tp / (tp + fp + fn + 1e-5)
    return iou


class Scores(object):

    def __init__(self, flag_ctp):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.LVO_confmat = {"TP":0,"TN":0,"FP":0,"FN":0}
        self.SVO_confmat = {"TP":0,"TN":0,"FP":0,"FN":0}

        self.coords,self.coords_LVO,self.coords_SVO = [],[],[]
        self.flag_ctp = flag_ctp

        self.patient_dice = []
        self.patient_iou = []
        self.patient_mcc = []
        self.patient_roi = []

        self.patient_dice_class = {}
        self.patient_iou_class = {}
        self.patient_mcc_class = {}

    def record(self, preds, label, query_id=""):
        assert len(torch.unique(preds)) < 3, "# of preds: {0} -- {1}".format(len(torch.unique(preds)), torch.unique(preds))

        tp = torch.sum((label == 1) * (preds == 1), dtype=torch.float32)
        tn = torch.sum((label == 0) * (preds == 0), dtype=torch.float32)
        fp = torch.sum((label == 0) * (preds == 1), dtype=torch.float32)
        fn = torch.sum((label == 1) * (preds == 0), dtype=torch.float32)
        roi = torch.sum((label == 1), dtype=torch.float32)

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-5)
        iou = tp / (tp + fp + fn + 1e-5)
        mcc = ((tn * tp) - (fp * fn)) / (torch.sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp)) + 1e-5)
        self.patient_dice.append(dice)
        self.patient_iou.append(iou)
        self.patient_mcc.append(mcc)
        self.patient_roi.append(roi)

        self.coords.append((roi.item(), dice))

        if self.flag_ctp:
            if "_00_" in query_id or "_01_" in query_id or "_20_" in query_id or "_21_" in query_id:
                if "LVO" not in self.patient_dice_class.keys(): self.patient_dice_class["LVO"] = []
                if "LVO" not in self.patient_iou_class.keys(): self.patient_iou_class["LVO"] = []
                if "LVO" not in self.patient_mcc_class.keys(): self.patient_mcc_class["LVO"] = []
                if "both" not in self.patient_dice_class.keys(): self.patient_dice_class["both"] = []
                if "both" not in self.patient_iou_class.keys(): self.patient_iou_class["both"] = []
                if "both" not in self.patient_mcc_class.keys(): self.patient_mcc_class["both"] = []
                self.patient_dice_class["LVO"].append(dice)
                self.patient_iou_class["LVO"].append(iou)
                self.patient_mcc_class["LVO"].append(mcc)
                self.patient_dice_class["both"].append(dice)
                self.patient_iou_class["both"].append(iou)
                self.patient_mcc_class["both"].append(mcc)
                self.coords_LVO.append((roi.item(), dice))
                self.LVO_confmat["TP"]+=tp
                self.LVO_confmat["FP"]+=fp
                self.LVO_confmat["TN"]+=tn
                self.LVO_confmat["FN"]+=fn
            elif "_02_" in query_id or "_22_" in query_id:
                if "SVO" not in self.patient_dice_class.keys(): self.patient_dice_class["SVO"] = []
                if "SVO" not in self.patient_iou_class.keys(): self.patient_iou_class["SVO"] = []
                if "SVO" not in self.patient_mcc_class.keys(): self.patient_mcc_class["SVO"] = []
                if "both" not in self.patient_dice_class.keys(): self.patient_dice_class["both"] = []
                if "both" not in self.patient_iou_class.keys(): self.patient_iou_class["both"] = []
                if "both" not in self.patient_mcc_class.keys(): self.patient_mcc_class["both"] = []
                self.patient_dice_class["SVO"].append(dice)
                self.patient_iou_class["SVO"].append(iou)
                self.patient_mcc_class["SVO"].append(mcc)
                self.patient_dice_class["both"].append(dice)
                self.patient_iou_class["both"].append(iou)
                self.patient_mcc_class["both"].append(mcc)
                self.coords_SVO.append((roi.item(), dice))
                self.SVO_confmat["TP"] += tp
                self.SVO_confmat["FP"] += fp
                self.SVO_confmat["TN"] += tn
                self.SVO_confmat["FN"] += fn

        self.TP += tp
        self.TN += tn
        self.FP += fp
        self.FN += fn

    def compute_dice(self,flag=""):
        if flag=="LVO": return 2 * self.LVO_confmat["TP"] / (2 * self.LVO_confmat["TP"] + self.LVO_confmat["FP"] + self.LVO_confmat["FN"])
        elif flag=="SVO": return 2 * self.SVO_confmat["TP"] / (2 * self.SVO_confmat["TP"] + self.SVO_confmat["FP"] + self.SVO_confmat["FN"])
        else: return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

    def compute_iou(self,flag=""):
        if flag=="LVO": return self.LVO_confmat["TP"] / (self.LVO_confmat["TP"] + self.LVO_confmat["FP"] + self.LVO_confmat["FN"])
        elif flag=="SVO": return self.SVO_confmat["TP"] / (self.SVO_confmat["TP"] + self.SVO_confmat["FP"] + self.SVO_confmat["FN"])
        else: return self.TP / (self.TP + self.FP + self.FN)

    def get_coords(self, flag, index):
        if flag=="LVO": return [el[index] for el in self.coords_LVO]
        elif flag=="SVO": return [el[index] for el in self.coords_SVO]
        else: return [el[index] for el in self.coords]


import os
import numpy as np
from saliency_metrics import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, cal_iou, Eval, Statistic
from PIL import Image
import torchvision.transforms as transforms
from multiprocessing import Pool, Lock, Manager, Process
from multiprocessing.managers import BaseManager
import multiprocessing

import torch
#
# from multiprocessing import set_start_method
#
# try:
#     set_start_method('forkserver', force=True)
#
# except RuntimeError:
#     pass

os.environ["CUDA_VISION_DEVICES"] = "2"

dataset_path = '..'  ##gt_path

dataset_path_pre = '..'  ##pre_salmap_path

test_datasets = ['test_data']  ##test_datasets_name

def parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--td',
                        type=str,
                        nargs='+',
                        default=[])
    parser.add_argument('--scale',
                        type=int,
                        default=1,
                        )
    args = parser.parse_args()
    return args

class eval_dataset:
    def __init__(self, image_root, gt_root):
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(image_root) if f.endswith('.png')]
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        # self.index = 0

    def load_data(self, index):
        # image = self.rgb_loader(self.images[self.index])
        # self.index = index
        image = self.binary_loader(os.path.join(self.image_root, self.img_list[index] + '.png'))
        gt = self.binary_loader(os.path.join(self.gt_root, self.img_list[index] + '.png'))
        w, h = image.size
        image = image.resize((w // args.scale, h // args.scale))
        gt = gt.resize((w // args.scale, h // args.scale))
        # self.index += 1
        if image.size != gt.size:
            x, y = gt.size
            image = image.resize((x, y))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0
        res = image
        res = np.array(res)
        if res.max() == res.min():
            res = res / 255
        else:
            res = (res - res.min()) / (res.max() - res.min())

        return torch.from_numpy(res), torch.from_numpy(gt)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


def run(index):
    print('predicting for %d / %d' % (index + 1, test_loader.size))
    res, gt = test_loader.load_data(index)
    device = torch.device('cuda')

    model = Eval().to(device)
    mae, iou = model(res, gt)
    metrics.update(mae, iou)


class Mymanager(BaseManager):
    pass


def Manager2():
    m = Mymanager()
    m.start()
    return m


Mymanager.register('cal_mae', cal_mae)
Mymanager.register('cal_fm', cal_fm)
Mymanager.register('cal_sm', cal_sm)
Mymanager.register('cal_em', cal_em)
Mymanager.register('cal_wfm', cal_wfm)
Mymanager.register('cal_iou', cal_iou)
Mymanager.register('Statistic', Statistic)


if __name__ == '__main__':

    # lock = Lock()
    args = parser()
    import datetime

    start = datetime.datetime.now()
    for dataset in args.td:
        print('================eval: '+dataset+'===============')
        sal_root = dataset_path_pre + '/eval/maps/F3Net/' + dataset
        gt_root = dataset_path + '/data/' + dataset + '/mask/'
        test_loader = eval_dataset(sal_root, gt_root)

        # instance share class
        manager = Manager2()

        metrics = manager.Statistic()
        f = open(dataset + '-eval.txt', 'w')

        p = Pool(multiprocessing.cpu_count())

        for i in range(test_loader.size):
            p.apply_async(run, args=(i, ))

        p.close()
        p.join()

        MAE, IOU = metrics.show()

        print('dataset: {} MAE: {:.4f} IOU: {:.4f}'.format(dataset, MAE,IOU))
        f.write(
            'dataset: {} MAE: {:.4f} IOU: {:.4f}'.format(dataset, MAE, IOU))
        f.close()
    print(datetime.datetime.now() - start)

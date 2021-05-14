import os
import numpy as np
from saliency_metrics import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, cal_iou
from PIL import Image
import torchvision.transforms as transforms
from multiprocessing import Pool, Lock, Manager, Process
from multiprocessing.managers import BaseManager
import multiprocessing

dataset_path = '..'  ##gt_path

dataset_path_pre = '..'  ##pre_salmap_path

test_datasets = ['DUT']  ##test_datasets_name


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
        scale = 1
        w, h = image.size
        image = image.resize((w // scale, h // scale))
        gt = gt.resize((w // scale, h // scale))

        # self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


def run(index, mae, sm, fm, em, wfm, iou):
    # with lock:
    print('predicting for %d / %d' % (index + 1, test_loader.size))
    sal, gt = test_loader.load_data(index)
    if sal.size != gt.size:
        x, y = gt.size
        sal = sal.resize((x, y))
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    gt[gt > 0.5] = 1
    gt[gt != 1] = 0
    res = sal
    res = np.array(res)
    if res.max() == res.min():
        res = res / 255
    else:
        res = (res - res.min()) / (res.max() - res.min())
    mae.update(res, gt)
    sm.update(res, gt)
    fm.update(res, gt)
    em.update(res, gt)
    wfm.update(res, gt)
    iou.update(res, gt)




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

if __name__ == '__main__':

    # lock = Lock()
    import datetime

    start = datetime.datetime.now()
    for dataset in test_datasets:
        sal_root = dataset_path_pre + '/eval/maps/F3Net/' + dataset
        gt_root = dataset_path + '/data/' + dataset + '/mask/'
        test_loader = eval_dataset(sal_root, gt_root)

        # instance share class
        manager = Manager2()
        mae = manager.cal_mae()
        fm = manager.cal_fm(test_loader.size)
        sm = manager.cal_sm()
        em = manager.cal_em()
        wfm = manager.cal_wfm()
        iou = manager.cal_iou()

        # mae,fm,sm,em,wfm= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm()
        f = open(dataset + '-eval.txt', 'w')

        p = Pool(multiprocessing.cpu_count())

        for i in range(test_loader.size):
            p.apply_async(run, args=(i, mae, fm, sm, em, wfm, iou))

        p.close()
        p.join()

        MAE = mae.show()
        maxf, meanf, _, _ = fm.show()
        sm = sm.show()
        em = em.show()
        wfm = wfm.show()
        IOU = iou.show()

        print('dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f} IOU: {:.4f}'.format(dataset,
                                                                                                           MAE,
                                                                                                           maxf,
                                                                                                           meanf,
                                                                                                           wfm, sm,
                                                                                                           em,IOU))
        f.write(
            'dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f} IOU: {:.4f}'.format(dataset,
                                                                                                         MAE, maxf,
                                                                                                         meanf, wfm,
                                                                                                         sm, em, IOU))
        f.close()
    print(datetime.datetime.now() - start)

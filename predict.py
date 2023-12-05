# coding=utf-8
import os
import torch.utils.data
from torch.utils.data import DataLoader
from data_utils import  TestDatasetFromFolder, calMetric_iou
from tqdm import tqdm
import argparse
import numpy as np
from models.mscanet.network import CDNet
from PIL import Image


parser = argparse.ArgumentParser(description='Test Change Detection Models')
parser.add_argument('--gpu_id', default="0,1,2,3", type=str, help='which gpu to run.')
parser.add_argument('--model_dir', default='epochs/netCD_epoch_100.pth', type=str)
parser.add_argument('--n_class', default=2, type=int, help='number of class')
parser.add_argument('--in_chan', default=3, type=int, help='channel of input image')
parser.add_argument('--batch_size', default=8, type=int, help='channel of input image')
parser.add_argument('--crop_size', default=512, type=int, help='channel of input image')
parser.add_argument('--path_img1', default='../Data/test/image1', type=str, help='whether used cbam trick')
parser.add_argument('--path_img2', default='../Data/test/image2', type=str, help='whether used cbam trick')
parser.add_argument('--path_lab', default='../Data/test/label', type=str, help='whether used cbam trick')
parser.add_argument('--save_dir', default='result/test/', type=str)

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(opt.save_dir):
    os.mkdir(opt.save_dir)

CDNet =  CDNet(img_size = opt.crop_size).to(device, dtype=torch.float)


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    netCD = torch.nn.DataParallel(CDNet, device_ids=range(torch.cuda.device_count()))

netCD.load_state_dict(torch.load(opt.model_dir))

netCD.eval()

if __name__ == '__main__':
    test_set = TestDatasetFromFolder(opt, opt.path_img1, opt.path_img2, opt.path_lab)
    test_loader = DataLoader(dataset=test_set, num_workers=24, batch_size=opt.batch_size, shuffle=True)
    test_bar = tqdm(test_loader)
    inter = 0
    unin = 0

    test_results = { 'batch_sizes': 0, 'IoU': 0, 'f1': 0}

    for image1, image2, label, image_name in test_bar:
        # print(image_name)
        test_results['batch_sizes'] += 1

        image1 = image1.to(device, dtype=torch.float)
        image2 = image2.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.float)

        output = netCD(image1, image2)

        label = torch.argmax(label, 1).unsqueeze(1)
        output = torch.argmax(output, 1).unsqueeze(1)

        for i in range(label.size()[0]):
            gt_value = label[i]
            prob = output[i]

            gt_value = (gt_value > 0).float()
            prob = (prob > 0).float()
            prob = prob.cpu().detach().numpy()
            gt_value = gt_value.cpu().detach().numpy()
            gt_value = np.squeeze(gt_value)
            result = np.squeeze(prob)

            intr, unn = calMetric_iou(gt_value, result)
            inter = inter + intr
            unin = unin + unn

            # loss for current batch before optimization
            test_results['IoU'] = (inter * 1.0 / unin)

            test_bar.set_description(
                desc='IoU: %.4f' % ( test_results['IoU'] ))

            result = Image.fromarray(result.astype('uint8'))
            result.save(opt.save_dir + image_name[i])


import os
import copy
import random
import argparse
import time
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import torch,sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.dirname(os.path.dirname(BASE_DIR)))
from data_controller_gray import SegDataset
from loss import Loss
from segnet import SegNet as segnet
import sys
sys.path.append("..")
from lib.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/home/ryan/BOP_dataset/new_data/train_pbr', help="dataset root dir (''YCB_Video Dataset'')")
parser.add_argument('--batch_size', default=1, help="batch size")
parser.add_argument('--n_epochs', default=50, help="epochs to train")
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help="learning rate")
parser.add_argument('--logs_path', default='logs/', help="path to save logs")
parser.add_argument('--model_save_path', default=BASE_DIR + '/trained_models', help="path to save models")
parser.add_argument('--log_dir', default=BASE_DIR + '/logs/', help="path to save logs")
parser.add_argument('--resume_model', default="model_1_0.05777042932518549.pth", help="resume model name")
opt = parser.parse_args()

if __name__ == '__main__':
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = SegDataset('train', opt.dataset_root, True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers))
    test_dataset = SegDataset('test', opt.dataset_root, False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=int(opt.workers))

    model = segnet(input_nbr=1, label_nbr=2)
    model = model.cuda()
    if not os.path.exists(opt.log_dir): os.mkdir(opt.log_dir)
    if opt.resume_model != '':
        checkpoint = torch.load('{0}/{1}'.format(opt.model_save_path, opt.resume_model))
        model.load_state_dict(checkpoint)
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
        print("load model!")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = Loss()
    best_val_cost = np.Inf
    st_time = time.time()

    for epoch in range(1, opt.n_epochs):
        model.train()
        train_all_cost = 0.0
        train_time = 0
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))

        for i, data in enumerate(dataloader, 0):
            rgb, target = data
            # print(rgb.shape)
            rgb, target = Variable(rgb).cuda(), Variable(target).cuda()
            semantic = model(rgb)
            optimizer.zero_grad()
            semantic_loss = criterion(semantic, target)

            if i % 20 != 0:
                count_num = i % 20
                pred = torch.argmax(semantic.view(1, 2, 1024, 1280).contiguous(), dim=1).detach().cpu().numpy()
                pred = np.squeeze(pred)
                save_img = Image.fromarray(np.uint8(pred) * 255)
                # print(np.max(pred))
                save_img.save((BASE_DIR + '/img_test/' + '{0}.png').format('%06d' % count_num))

            train_all_cost += semantic_loss.item()
            # train_all_cost += semantic_loss
            semantic_loss.backward()
            optimizer.step()
            logger.info('Train time {0} Batch {1} CEloss {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), train_time, semantic_loss.item()))
            if train_time != 0 and train_time % 10 == 0:
                torch.save(model.state_dict(), os.path.join(opt.model_save_path, 'model_current.pth'))
            train_time += 1

        train_all_cost = train_all_cost / train_time
        logger.info('Train Finish Avg CEloss: {0}'.format(train_all_cost))
        if train_all_cost <= best_val_cost:
            best_val_cost = train_all_cost
            torch.save(model.state_dict(), os.path.join(opt.model_save_path, 'model_{}_{}.pth'.format(epoch, train_all_cost)))

        # model.eval()
        # test_all_cost = 0.0
        # test_time = 0
        # logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        # logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        # for j, data in enumerate(test_dataloader, 0):
        #     rgb, target = data
        #     rgb, target = Variable(rgb).cuda(), Variable(target).cuda()
        #     # print(rgb)
        #     semantic = model(rgb)
        #     # pred = torch.argmax(semantic.view(1, 22, 480, 640).contiguous(), dim=1).detach().cpu().numpy()
        #     semantic_loss = criterion(semantic, target)
        #     if j % 20 != 0:
        #         count_num = j % 20
        #         pred = torch.argmax(semantic.view(1, 2, 1024, 1280).contiguous(), dim=1).detach().cpu().numpy()
        #         pred = np.squeeze(pred)
        #         save_img = Image.fromarray(np.uint8(pred) * 255)
        #         # print(np.max(pred))
        #         save_img.save((BASE_DIR + '/img_test/' + '{0}.png').format('%06d' % count_num))
        #     test_all_cost += semantic_loss.item()
        #     # test_all_cost += semantic_loss
        #     test_time += 1
        #     logger.info('Test time {0} Batch {1} CEloss {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_time, semantic_loss.item()))
        #
        # test_all_cost = test_all_cost / test_time
        # logger.info('Test Finish best_val_cost: {0}'.format(best_val_cost))
        # logger.info('Test Finish Avg CEloss: {0}'.format(test_all_cost))
        #
        # if test_all_cost <= best_val_cost:
        #     best_val_cost = test_all_cost
        #     torch.save(model.state_dict(), os.path.join(opt.model_save_path, 'model_{}_{}.pth'.format(epoch, test_all_cost)))
        #     print('----------->BEST SAVED<-----------')

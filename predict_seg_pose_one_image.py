import os
import argparse
import sys
import numpy as np
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import copy
from torch.autograd import Variable
import torch.utils.data
from Seg_Pose.own_dataset_gray_one_image import Seg_PoseDataset_one_image as sp_Dataset
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
import sys
sys.path.append("..")

testdataset = 0
estimator = 0
refiner = 0
num_points = 500
num_objects = 1

def load_net_model():
        parser = argparse.ArgumentParser()
        parser.add_argument('--seg_model', type=str,
                            default='D:/DenseFusion_1.0_20210308/Seg_Pose/Seg/trained_models/model_2_0.0033125885542725133.pth',
                            help='resume Seg model')
        parser.add_argument('--model', type=str,
                            default='D:/DenseFusion_1.0_20210308/Seg_Pose/Pose/trained_models/pose_model_2_0.09233990000864985.pth',
                            help='resume PoseNet model')
        parser.add_argument('--refine_model', type=str,
                            default='D:/DenseFusion_1.0_20210308/Seg_Pose/Pose/trained_models/pose_refine_model_4_0.06547849291308346.pth',
                            help='resume PoseRefineNet model')
        opt = parser.parse_args()

        global estimator
        estimator = PoseNet(num_points=num_points, num_obj=num_objects)
        estimator.cuda()

        global refiner
        refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
        refiner.cuda()

        estimator.load_state_dict(torch.load(opt.model))
        refiner.load_state_dict(torch.load(opt.refine_model))
        estimator.eval()
        refiner.eval()

        global testdataset
        testdataset = sp_Dataset(opt.seg_model, num_points, False, 0.0, True)

def Csv_6D_pose(rgb_img, depth_img):
        iteration = 4
        bs = 1
        # knn = KNearestNeighbor(1)
        points, choose, img = testdataset.getitem_by_array(rgb_img, depth_img)
        if choose.ndim < 3:
                return []
        # print("choose.ndim =", choose.ndim)
        obj_id = torch.LongTensor([0]).unsqueeze(0)

        points, choose, img, obj_id = Variable(points).cuda(),  Variable(choose).cuda(), Variable(img).cuda(), Variable(obj_id).cuda()


        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, obj_id)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)

        for ite in range(0, iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t

                new_points = torch.bmm((points - T), R).contiguous()
                pred_r, pred_t = refiner(new_points, emb, obj_id)
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).cpu().data.numpy()
                my_t_2 = pred_t.view(-1).cpu().data.numpy()
                my_mat_2 = quaternion_matrix(my_r_2)
                my_mat_2[0:3, 3] = my_t_2

                my_mat_final = np.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_r_final[0:3, 3] = 0
                my_r_final = quaternion_from_matrix(my_r_final, True)
                my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                my_pred = np.append(my_r_final, my_t_final)
                my_r = my_r_final
                my_t = my_t_final

        print("final prediction: quaternion + translation")
        my_pred = np.asarray(my_pred, dtype='float')
        print(list(my_pred))
        return list(my_pred)
if __name__ == '__main__':
        path_test = os.getcwd()
        load_net_model()
        rgb_path = "D:/DenseFusion_1.0_20210308/Seg_Pose/000015.jpg"
        depth_path = "D:/DenseFusion_1.0_20210308/Seg_Pose/000015.png"
        rgb_img = Image.open(rgb_path).convert("L")
        depth_img = Image.open(depth_path)
        for i in range(4):
                Csv_6D_pose(rgb_img, depth_img)
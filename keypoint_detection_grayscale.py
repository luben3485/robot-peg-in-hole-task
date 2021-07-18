#! /usr/bin/env python
import mankey.network.inference_grayscale as inference_grayscale
from mankey.utils.imgproc import PixelCoord
import argparse
import os
import numpy as np
import cv2
import open3d as o3d
from bbox_detector import BboxDetection

num = '000047'
parser = argparse.ArgumentParser()
parser.add_argument('--net_path', type=str,
                    #default='/home/luben/data/trained_model/keypoint/mug/checkpoint-135.pth',
                    default='/Users/cmlab/robot-peg-in-hole-task/mankey/experiment/ckpnt_box_fix_2_grayscale/checkpoint-100.pth',
                    help='The absolute path to network checkpoint')
parser.add_argument('--cv_rgb_path', type=str,
                    #default='/home/luben/robotic-arm-task-oriented-manipulation/test_data/000000_rgb.png',
                    default='/Users/cmlab/data/pdc/logs_proto/box_insertion_fix_2_test/processed/images/'+ num + '_rgb.png',
                    help='The absolute path to rgb image')
parser.add_argument('--cv_depth_path', type=str,
                    #default='/home/luben/robotic-arm-task-oriented-manipulation/test_data/000000_depth.png',
                    default='/Users/cmlab/data/pdc/logs_proto/box_insertion_fix_2_test/processed/images/'+ num +'_depth.png',
                    help='The absolute path to depth image')



class KeypointDetection(object):

    def __init__(self, network_ckpnt_path):

        # The network
        assert os.path.exists(network_ckpnt_path)
        self._network, self._net_config = inference_grayscale.construct_resnet_nostage(network_ckpnt_path)
        
    def inference(self, bbox, cv_rgb=None, cv_depth=None ,cv_rgb_path='', cv_depth_path=''):
        if cv_rgb_path != '':
            cv_rgb = cv2.imread(cv_rgb_path, cv2.IMREAD_COLOR)
        if cv_depth_path != '':
            cv_depth = cv2.imread(cv_depth_path, cv2.IMREAD_ANYDEPTH)
       	camera_keypoint, keypointxy_depth_realunit = self.process_raw(cv_rgb, cv_depth, bbox)
        camera_keypoint = camera_keypoint.T  # shape[0]: n sample, shape[1]: xyz
        keypointxy_depth_realunit = keypointxy_depth_realunit.T
        return camera_keypoint, keypointxy_depth_realunit

    def process_raw(
            self,
            cv_rgb,  # type: np.ndarray
            cv_depth,  # type: np.ndarray
            bbox, #type: np.ndarray  [x,y,w,h]
    ):  # type: (np.ndarray, np.ndarray, np.ndarray [x_min,y_min,x_max,y_max]) -> np.ndarray
        # Parse the bounding box
        top_left, bottom_right = PixelCoord(), PixelCoord()
        top_left.x = bbox[0]
        top_left.y = bbox[1]
        bottom_right.x = bbox[2]
        bottom_right.y = bbox[3]

        # Perform the inference
        imgproc_out = inference_grayscale.proc_input_img_raw(
            cv_rgb, cv_depth,
            top_left, bottom_right)
        keypointxy_depth_scaled = inference_grayscale.inference_resnet_nostage(self._network, imgproc_out)
        keypointxy_depth_realunit = inference_grayscale.get_keypoint_xy_depth_real_unit(keypointxy_depth_scaled)
        _, camera_keypoint = inference_grayscale.get_3d_prediction(
            keypointxy_depth_realunit,
            imgproc_out.bbox2patch)

        return camera_keypoint, keypointxy_depth_realunit

    def visualize_2d(self,bbox, keypoints, cv_rgb=None, cv_rgb_path=''):
        # keypoints : list, shape:(n,x,y,d)
        if cv_rgb_path != '':
            img = cv2.imread(cv_rgb_path, cv2.IMREAD_COLOR)
        else:
            img = cv_rgb.copy()

        for idx, keypoint in enumerate(keypoints):
            x = int(keypoint[0])
            y = int(keypoint[1])
            if idx % 2 == 0:
                cv2.circle(img, (x, y), 3, (0, 0, 255), 1)
            else:
                cv2.circle(img, (x, y), 3, (255, 0, 0), 1)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        # v-rep -> (rotate 180) -> input of keypoint detection network
        # rotate 180 for recovery
        (h, w)= img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        img = cv2.warpAffine(img, M, (w, h))

        cv2.imshow('visualize keypoint', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualize(self, keypoints, cv_rgb=None, cv_depth=None, cv_rgb_path='', cv_depth_path=''):

        vis_list = []
        if cv_rgb_path != '':
            cv_rgb = cv2.imread(cv_rgb_path, cv2.IMREAD_COLOR)
            #print(cv_rgb.shape)
        if cv_depth_path != '':
            cv_depth = cv2.imread(cv_depth_path, cv2.IMREAD_ANYDEPTH)

        color = o3d.geometry.Image(cv_rgb)
        depth = o3d.geometry.Image(cv_depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)

        pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
                "config/camera_intrinsic.json")
        #print(pinhole_camera_intrinsic.intrinsic_matrix)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, pinhole_camera_intrinsic)
        print('pcd:',pcd)
        vis_list.append(pcd)
        #xyz = np.asarray(pcd.points)
        #print(xyz.shape)
        #print(xyz)
        #o3d.io.write_point_cloud("mug.ply", pcd)
        #with open('keypoint.txt', 'w') as f:
        #    for keypoint in keypoints:
        #        f.write(' '.join(['%.8f' % k for k in keypoint]))
        #        f.write('\n')

        for keypoint in keypoints:
            keypoints_coords \
                = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1, origin=[keypoint[0], keypoint[1], keypoint[2]])
            vis_list.append(keypoints_coords)
        o3d.visualization.draw_geometries(vis_list)

    
def main(netpath, rgb, depth):
    #rgb = '/Users/cmlab/robot-peg-in-hole-task/rgb.png'
    #depth = '/Users/cmlab/robot-peg-in-hole-task/depth.png'
    kp_detection = KeypointDetection(netpath)
    bbox_detection = BboxDetection('/Users/cmlab/robot-peg-in-hole-task/mrcnn/ckpnt_box_insertion_fix_3')
    # 6007
    #bbox = np.array([93, 72, 222, 197])
    #bbox = np.array([0, 0, 255, 255])
    bbox = bbox_detection.predict(cv_rgb_path=rgb)
    print(bbox)
    camera_keypoint, keypointxy_depth_realunit = kp_detection.inference(cv_rgb_path=rgb, cv_depth_path=depth, bbox=bbox)
    print(camera_keypoint)
    #print('xyd',keypointxy_depth_realunit)
    kp_detection.visualize(cv_rgb_path=rgb, cv_depth_path=depth, keypoints=camera_keypoint)
    kp_detection.visualize_2d(bbox=bbox, cv_rgb_path=rgb, keypoints=keypointxy_depth_realunit)


if __name__ == '__main__':
    
    args = parser.parse_args()
    net_path = args.net_path
    cv_rgb_path = args.cv_rgb_path
    cv_depth_path = args.cv_depth_path
    main(netpath=net_path, rgb=cv_rgb_path, depth=cv_depth_path)

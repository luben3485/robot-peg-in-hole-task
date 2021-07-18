# simExtRemoteApiStart(19999)

try:
    import env.vrep as vrep
    #import vrep
except Exception as e:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('Error: %s'%str(e))

import time
import numpy as np
import cv2
import math
import time
from transforms3d.quaternions import mat2quat

#from env.gdn_grasp.grasp_candidate import Grasper

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    _EPS = np.finfo(float).eps * 4.0
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

class SingleRoboticArm():
    def __init__(self):
        print ('Program started')
        vrep.simxFinish(-1) # just in case, close all opened connections

        while True:
            self.clientID = vrep.simxStart('127.0.0.1',19997,True,True,-500000,5) # Connect to V-REP
            if self.clientID != -1:
                print ('Connected to remote API server')
                returnCode = vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_oneshot)
                break
            else:
                print ('Failed connecting to remote API server')


        time.sleep(1)
        self.emptyBuff = bytearray()

        sim_ret, self.robot_handle = vrep.simxGetObjectHandle(self.clientID, "UR5#", vrep.simx_opmode_oneshot_wait)
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.clientID, "Vision_sensor", vrep.simx_opmode_oneshot_wait)
        
        

        self.original_matrix = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],np.float)
        self.original_matrix_gdn = np.array([[-1,0,0,0],
                                             [0,-1,0,0],
                                             [0,0,1,0],
                                             [0,0,0,1]],np.float) 
        self.focal_length = 309.019
        self.principal = 128

    def get_camera_matrix(self, cam_name='Vision_sensor'):
        #res,retInts,self.robot_matrix,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[robot_handle],[],[],emptyBuff,vrep.simx_opmode_oneshot_wait)
        sim_ret, cam_handle = vrep.simxGetObjectHandle(self.clientID, cam_name, vrep.simx_opmode_oneshot_wait)
        res,retInts,self.cam_matrix,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[cam_handle],[],[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
        sim_ret, self.cam_pos = vrep.simxGetObjectPosition(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        sim_ret, self.cam_quat = vrep.simxGetObjectQuaternion(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        self.cam_matrix = np.array(self.cam_matrix,dtype=np.float64).reshape(3,4)
        
        #print('='*20+'\ncam matrix')
        #print(cam_matrix)
        return self.cam_matrix
    
    def get_camera_pos(self, cam_name="Vision_sensor"):  
        sim_ret, cam_handle = vrep.simxGetObjectHandle(self.clientID, cam_name, vrep.simx_opmode_oneshot_wait)
        sim_ret, cam_pos = vrep.simxGetObjectPosition(self.clientID, cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        sim_ret, cam_quat = vrep.simxGetObjectQuaternion(self.clientID, cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        res,retInts,cam_matrix,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[cam_handle],[],[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
        
        return cam_pos, cam_quat, cam_matrix
    
    def get_intrinsic_matrix(self):
        intrinsic_matrix = np.array([[self.focal_length,0, self.principal],[0, self.focal_length, self.principal],[0,0,1]],dtype=np.float64)
        #print('='*20+'\nintrinsic matrix')
        #print(intrinsic)
        return intrinsic_matrix
        
    def get_rgb(self,cam_name= "Vision_sensor"):
        # RGB Info
        sim_ret, cam_handle = vrep.simxGetObjectHandle(self.clientID, cam_name, vrep.simx_opmode_oneshot_wait)
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.clientID, cam_handle, 0, vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)
        color_img[color_img < 0] += 255
        color_img = cv2.flip(color_img,0).astype(np.uint8)

        #color_img = color_img.astype(np.uint8)
        color_img = cv2.cvtColor(color_img,cv2.COLOR_RGB2BGR)

        return color_img

    def get_depth(self, cam_name="Vision_sensor", near_plane=0.02, far_plane= 0.5):
        # Depth Info
        sim_ret, cam_handle = vrep.simxGetObjectHandle(self.clientID, cam_name, vrep.simx_opmode_oneshot_wait)
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.clientID, cam_handle, vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = cv2.flip(depth_img,0).astype(np.float)

        depth_img = (far_plane - near_plane) * depth_img + near_plane
        return depth_img.astype(np.float64)
    
    def get_mask(self,cam_name="Vision_sensor"):
        color_img = self.get_rgb(cam_name=cam_name)
        img_gray = cv2.cvtColor(color_img,cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(img_gray,127,255,0)
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
        masks = []
        for i in range(len(contours)-1):
            mask = np.zeros((color_img.shape[0],color_img.shape[1]), np.uint8)
            cv2.drawContours(mask,[contours[i]], -1,255,-1)
            masks.append(mask)
        return np.array(masks)

    def visualize_image(self, mask=None, depth=None, rgb=None):

        if mask is not None:
            cv2.imshow("mask",mask)
        if depth is not None:
            depth = depth - np.min(depth)
            depth /= np.max(depth)
            depth = (depth*255.0).astype(np.uint8)
            cv2.imshow("depth", depth)
        if rgb is not None:
            cv2.imshow("rgb", rgb)
        cv2.waitKey(6000)
    
    def get_grasping_candidate(self, depth, extrinsic_matrix, ang_face, ang_bottom):
        
        g = Grasper()
        pcd = g.build_point_cloud(depth,self.get_intrinsic_matrix(), extrinsic_matrix).astype(np.float32) # (H,W,3)
        #np.save('dump.npy', pcd.reshape(-1, 3))
        #grasp_list = g.get_grasping_candidates(pcd,mask.astype(np.bool)) # (n, 3, 4)
        grasp_list = g.get_grasping_candidates(pcd)  # (n, 3, 4)
        grasp_list_filter = g.grasping_filter(grasp_list, ang_face, ang_bottom) # (n, 3, 4)
        
        return grasp_list_filter
    

    def naive_grasp_detection(self, rgb_img, depth_img, cam_name='Vision_sensor', show_bbox = True):

        img_gray = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(img_gray,127,255,0)
        im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
        rgb_img_show = rgb_img.copy()

        grasp_list = []
        for i in range(len(contours)-1):
            cnt = contours[i]
            rect = cv2.minAreaRect(cnt)    
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(rgb_img_show,[box],0,(0,0,255),2)
            
            pos_x = int((box[0][0]+box[2][0])/2)
            pos_y = int((box[0][1]+box[2][1])/2)
            pos_z = np.min(depth_img)
            angle = -1*rect[2]/180*math.pi

            z_offset = 0.02
            pos_x_cam = -1*pos_z*(pos_x-self.principal)/self.focal_length
            pos_y_cam =-1*pos_z*(pos_y-self.principal)/self.focal_length
            pos_cam = np.array([pos_x_cam,pos_y_cam,pos_z,1.0],np.float64).reshape(4,1)
            cam_matrix = self.get_camera_matrix(cam_name=cam_name)
            pos_world = np.dot(cam_matrix,pos_cam)

            pos_world[2,0] = pos_world[2,0] - z_offset

            grasp_matrix = np.array([[math.cos(angle),-math.sin(angle),0,pos_world[0,0]],
                    [math.sin(angle),math.cos(angle),0,pos_world[1,0]],
                    [0,0,1,pos_world[2,0]],
                    [0,0,0,1]],np.float64)
            grasp_list.append(grasp_matrix)

        if show_bbox:
            cv2.imshow('bbox',rgb_img_show)
            cv2.waitKey(1000)
        
        return grasp_list

    def naive_angle_grasp(self, rgb_img, depth_img, cam_name='Vision_sensor', show_bbox=False):
        self.enableIK(1)

        img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        rgb_img_show = rgb_img.copy()

        grasp_list = []
        for i in range(len(contours) - 1):
            cnt = contours[i]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(rgb_img_show, [box], 0, (0, 0, 255), 2)
            print(box)
            center_x = int((box[0][0] + box[2][0]) / 2)
            center_y = int((box[0][1] + box[2][1]) / 2)
            center_z = np.min(depth_img)
            pos_z = np.min(depth_img)

            #angle = -1 * rect[2] / 180 * math.pi
            edge_length = []
            for i in range(4):
                a = (i + 1) % 4
                b = i
                dis = ((box[a][0] - box[b][0]) ** 2 + (box[a][1] - box[b][1]) ** 2) ** 0.5
                edge_length.append(dis)
            print(rect[2])
            #left tilt
            if edge_length[0] < edge_length[1]:
                angle = (90 - rect[2]) / 180 * math.pi
                pos_x = int((box[2][0] + box[3][0]) / 2)
                pos_y = int((box[2][1] + box[3][1]) / 2)
            # right tilt
            else:
                angle = -1 * rect[2] / 180 * math.pi
                pos_x = int((box[0][0] + box[3][0]) / 2)
                pos_y = int((box[0][1] + box[3][1]) / 2)

            cv2.circle(rgb_img_show, (pos_x, pos_y),1,(0,255,0),4)
            cv2.circle(rgb_img_show, (center_x, center_y), 1, (255, 0, 0), 4)
            z_offset = 0.021
            center_x_cam = -1 * center_z * (center_x - self.principal) / self.focal_length
            center_y_cam = -1 * center_z * (center_y - self.principal) / self.focal_length
            center_cam = np.array([center_x_cam, center_y_cam, center_z, 1.0], np.float64).reshape(4, 1)

            pos_x_cam = -1 * pos_z * (pos_x - self.principal) / self.focal_length
            pos_y_cam = -1 * pos_z * (pos_y - self.principal) / self.focal_length
            pos_cam = np.array([pos_x_cam, pos_y_cam, pos_z, 1.0], np.float64).reshape(4, 1)

            cam_matrix = self.get_camera_matrix(cam_name=cam_name)
            pos_world = np.dot(cam_matrix, pos_cam)
            center_world = np.dot(cam_matrix, center_cam)

            rect_dir = np.array(pos_world - center_world)

            pos_world -= 0.4 * rect_dir

            rect_dir = rect_dir / np.linalg.norm(rect_dir)
            gripper_rot_dir = np.array([-1 * rect_dir[1], rect_dir[0], rect_dir[2]])
            pos_world[2, 0] = pos_world[2, 0] - z_offset

            grasp_matrix = np.array([[math.cos(angle), -math.sin(angle), 0, pos_world[0, 0]],
                                     [math.sin(angle), math.cos(angle), 0, pos_world[1, 0]],
                                     [0, 0, 1, pos_world[2, 0]],
                                     [0, 0, 0, 1]], np.float64)

            grasp_matrix = np.dot(grasp_matrix, self.original_matrix)

            #pre
            grasp_matrix[2:3] += 0.1
            self.movement(grasp_matrix)

            #grasp
            grasp_matrix[2:3] -= 0.1
            self.movement(grasp_matrix)

            #rot
            degree = 86
            w = math.cos(math.radians(degree / 2))
            x = math.sin(math.radians(degree / 2)) * gripper_rot_dir[0]
            y = math.sin(math.radians(degree / 2)) * gripper_rot_dir[1]
            z = math.sin(math.radians(degree / 2)) * gripper_rot_dir[2]
            gripper_rot_quat = [w, x, y, z]
            rot_pose = quaternion_matrix(gripper_rot_quat)
            rot_matrix = np.dot(rot_pose[:3, :3], grasp_matrix[:3, :3])
            grasp_matrix[:3, :3] = rot_matrix

            self.movement(grasp_matrix)
            self.open_gripper(False)

            grasp_matrix[2,3] += 0.1
            self.movement(grasp_matrix)

            step_3_pose = self.get_object_matrix(obj_name='stage_3_pose')
            self.movement(step_3_pose)

        if show_bbox:
            cv2.imshow('bbox', rgb_img_show)
            cv2.waitKey(5000)

    def gdn_demo_grasp(self):
        self.enableIK(1)

        cam_name = 'vision_sensor_gdn'
        rgb = self.get_rgb(cam_name=cam_name)
        depth = self.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=1.5)
        # masks = self.get_mask(cam_name=cam_name)
        extrinsic = self.get_camera_matrix(cam_name=cam_name)
        self.visualize_image(mask=None, depth=depth, rgb=None)
        grasp_list = self.get_grasping_candidate(depth, extrinsic, 90, 90)
        grasp_matrix = grasp_list[0]
        self.run_single_grasp(grasp_matrix, use_gdn=True)

        step_2_pose = self.get_object_matrix(obj_name='stage_2_pose')
        step_2_rot = step_2_pose[:3, :3]
        grasp_matrix[:3, :3] = step_2_rot
        grasp_matrix[2, 3] += 0.2
        self.movement(grasp_matrix)
        self.movement(step_2_pose)
        self.open_gripper(True)
        step_3_pose = self.get_object_matrix(obj_name='stage_3_pose')
        self.movement(step_3_pose)

    def run_single_grasp(self, grasp, use_gdn=True):
        self.enableIK(1)

        if use_gdn:
            grasp_matrix = np.dot(grasp, self.original_matrix_gdn)
        else:
            grasp_matrix = np.dot(grasp, self.original_matrix)

        # visualize the grasping pose
        rot_matrix = grasp_matrix[:3, :3]
        quat = mat2quat(rot_matrix)
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        self.set_object_quat('staples_pose', quat)
        self.set_object_position('staples_pose', grasp_matrix[:3, 3])

        send_matrix = list(grasp_matrix.flatten())[0:12]
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',
                                                                                    vrep.sim_scripttype_childscript,
                                                                                    'GraspMovement', [], send_matrix,
                                                                                    [], self.emptyBuff,
                                                                                    vrep.simx_opmode_oneshot_wait)
        running = True
        while running:
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',
                                                                                             vrep.sim_scripttype_childscript,
                                                                                             'isRunning', [], [],
                                                                                             ['UR5_GraspMatrix'],
                                                                                             self.emptyBuff,
                                                                                             vrep.simx_opmode_oneshot_wait)
            if retInts[0] == 0:
                running = False

    def run_grasp(self, grasp_list, num, use_gdn = True):
        self.enableIK(1)

        if len(grasp_list) == 0:
            print('No any grasp detected!')
        grasp_iter = min(num, len(grasp_list))
        for i in range(grasp_iter):
            if use_gdn:
                grasp_matrix = np.dot(grasp_list[i], self.original_matrix_gdn)
            else:
                grasp_matrix = np.dot(grasp_list[i], self.original_matrix)

            #visualize the grasping pose
            rot_matrix = grasp_matrix[:3, :3]
            quat = mat2quat(rot_matrix)
            quat = np.array([quat[1], quat[2], quat[3], quat[0]])
            self.set_object_quat('staples_pose', quat)
            self.set_object_position('staples_pose', grasp_matrix[:3,3])

            send_matrix = list(grasp_matrix.flatten())[0:12]
            res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'GraspMovement',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait) 
            running = True
            while running:
                res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_GraspMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
                if retInts[0] == 0:
                    running = False



    # this function is for main_vrep_angle_insertion and collect_insertion_data_eye_reverse
    def gt_run_grasp(self, grasp_matrix):
        self.enableIK(1)
        send_matrix = list(grasp_matrix.flatten())[0:12]
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',vrep.sim_scripttype_childscript,'GraspMovement', [], send_matrix,[], self.emptyBuff,vrep.simx_opmode_oneshot_wait)
        running = True
        while running:
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning', [], [],['UR5_GraspMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
            if retInts[0] == 0:
                running = False

    
    def get_object_matrix(self, obj_name):
        #self.enableIK(1)
        sim_ret, object_handle = vrep.simxGetObjectHandle(self.clientID, obj_name, vrep.simx_opmode_oneshot_wait)
        res,retInts, object_matrix,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[object_handle],[],[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
        object_matrix = np.array(object_matrix+[0,0,0,1],dtype=np.float64).reshape(4,4)
        return object_matrix

    def get_object_quat(self, obj_name):
        self.enableIK(1)
        sim_ret, object_handle = vrep.simxGetObjectHandle(self.clientID, obj_name, vrep.simx_opmode_oneshot_wait)
        ret, object_quat = vrep.simxGetObjectQuaternion(self.clientID, object_handle, -1, vrep.simx_opmode_oneshot_wait)
        return object_quat

    def set_object_position(self, obj_name, obj_position):
        sim_ret, obj_handle = vrep.simxGetObjectHandle(self.clientID, obj_name, vrep.simx_opmode_oneshot_wait)
        sim_ret = vrep.simxSetObjectPosition(self.clientID, obj_handle, -1, obj_position, vrep.simx_opmode_oneshot_wait)

    def set_object_quat(self, obj_name, obj_quat):
        sim_ret, obj_handle = vrep.simxGetObjectHandle(self.clientID, obj_name, vrep.simx_opmode_oneshot_wait)
        sim_ret = vrep.simxSetObjectQuaternion(self.clientID, obj_handle, -1, obj_quat, vrep.simx_opmode_oneshot_wait)

    ''' API has no function of 'vrep.simxSetObjectMatrix'
    def set_object_pose(self, obj_name, obj_pose):
        obj_pose = obj_pose[:3].reshape(12)
        sim_ret, obj_handle = vrep.simxGetObjectHandle(self.clientID, obj_name, vrep.simx_opmode_oneshot_wait)
        sim_ret = vrep.simxSetObjectMatrix(self.clientID, obj_handle, -1, obj_pose, vrep.simx_opmode_oneshot_wait)
    '''
    def open_gripper(self, mode):
        mode = int(mode)
        send_matrix = [mode]
        res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'Open',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
        running = True
        while running:
            res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_Open'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
            if retInts[0] == 0:
                running = False

    def movement(self, pose_matrix):
        self.enableIK(1)
        send_matrix = list(pose_matrix.flatten())[0:12]
        res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'Movement',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait) 
        running = True
        while running:
            res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_MovementMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
            if retInts[0] == 0:
                running = False

    def insertion(self, pose_matrix):
        self.enableIK(1)
        send_matrix = list(pose_matrix.flatten())[0:12]
        res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'Insertion',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait) 
        running = True
        while running:
            res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_InsertionMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
            if retInts[0] == 0:
                running = False
        

    def gt_peg_in_hole_servo(self, target_name, object_name, err_tolerance=0.3, alpha_err=0.9, alpha_target=0.9):
        self.enableIK(1)

        filter_robot_pose = self.get_object_matrix('UR5_ikTip')
        err_rolling, err_size_rolling = None, err_tolerance * 10
        cnt = 0
        while err_size_rolling > err_tolerance:
            print('========cnt:' + str(cnt) + '=========\n')
            cnt += 1
            robot_pose = self.get_object_matrix('UR5_ikTip')
            peg_keypoint_pose = self.get_object_matrix(object_name)
            hole_keypoint_pose = self.get_object_matrix(target_name)
            object_keypoint = peg_keypoint_pose[:3, 3]
            target_keypoint = hole_keypoint_pose[:3, 3]
            print('hole', target_keypoint)
            err = target_keypoint - object_keypoint
            # err*= 0.1
            print('err vector', err)

            # robot_pose[1,3] += (-err[0])
            # robot_pose[2,3] += (-err[1])
            # robot_pose[2,3] += err[1]
            robot_pose[:2, 3] += err[:2]

            print('unfiltered robot pose', robot_pose[:3, 3])
            filter_robot_pose = alpha_target * filter_robot_pose + (1 - alpha_target) * robot_pose
            print('filtered robot pose', filter_robot_pose[:3, 3])
            # robot moves to filter_robot_pose
            self.movement(filter_robot_pose)
            # self.movement(robot_pose)
            if err_rolling is None:
                err_rolling = err
            err_rolling = alpha_err * err_rolling + (1 - alpha_err) * err
            err_size_rolling = alpha_err * err_size_rolling + (1 - alpha_err) * np.linalg.norm(err_rolling)

        # place
        robot_pose = self.get_object_matrix('UR5_ikTip')
        peg_keypoint_pose = self.get_object_matrix(object_name)
        hole_keypoint_pose = self.get_object_matrix(target_name)
        err = target_keypoint - object_keypoint
        err *= 0.9
        print('ferr', err)
        robot_pose[2, 3] += err[2]
        # robot_pose[2,3] -= 0.9
        self.insertion(robot_pose)

    def finish(self):
        vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_oneshot_wait)
        vrep.simxGetPingTime(self.clientID)
        vrep.simxFinish(self.clientID)
        print('Stop server!')

    def get_joint_position(self):
        res, retInts, robotCurrentConfig, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID,
                                                                                              'RemoteAPI',
                                                                                              vrep.sim_scripttype_childscript,
                                                                                              'getRobotState',
                                                                                              [self.robot_handle], [], [],
                                                                                              self.emptyBuff,
                                                                                          vrep.simx_opmode_oneshot_wait)
        return robotCurrentConfig
    def enableIK(self, mode):
        mode = int(mode)
        send_matrix = [mode]
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',
                                                                                vrep.sim_scripttype_childscript, 'IK',
                                                                                [], send_matrix, [], self.emptyBuff,
                                                                                vrep.simx_opmode_oneshot_wait)

        running = True
        while running:
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',
                                                                                         vrep.sim_scripttype_childscript,
                                                                                         'isRunning', [], [],
                                                                                         ['UR5_IK'],
                                                                                         self.emptyBuff,
                                                                                         vrep.simx_opmode_oneshot_wait)
            if retInts[0] == 0:
                running = False

    def compute_path_from_joint_space(self, jointConfig, interpolation_states):
        approachVector = [0, 0, 0]  # often a linear approach is required. This should also be part of the calculations when selecting an appropriate state for a given pose
        maxConfigsForDesiredPose = 100  # we will try to find 10 different states corresponding to the goal pose and order them according to distance from initial state
        maxTrialsForConfigSearch = 300  # a parameter needed for finding appropriate goal states
        searchCount = 8  # how many times OMPL will run for a given task
        minConfigsForPathPlanningPath = interpolation_states  # interpolation states for the OMPL path
        minConfigsForIkPath = 100  # interpolation states for the linear approach path
        collisionChecking = 1  # whether collision checking is on or off
        inInts = [self.robot_handle, collisionChecking, minConfigsForPathPlanningPath, searchCount]

        res, retInts, robotCurrentConfig, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID,
                                                                                          'RemoteAPI',
                                                                                          vrep.sim_scripttype_childscript,
                                                                                          'getRobotState',
                                                                                          [self.robot_handle], [], [],
                                                                                          self.emptyBuff,
                                                                                          vrep.simx_opmode_oneshot_wait)
        print('robotCurrentConfig', robotCurrentConfig)
        tryConfig_1 = [-1.0110, -1.1291, 0.7916, 0.3375, 2.1395, 0.7224]
        inFloats = robotCurrentConfig + jointConfig

        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',
                                                                            vrep.sim_scripttype_childscript,
                                                                            'findPath_goalIsState', inInts, inFloats,
                                                                            [], self.emptyBuff,
                                                                            vrep.simx_opmode_oneshot_wait)
        return path

    def run_through_all_path(self, path):
        if len(path) > 0:
            for i in range(int(len(path) / 6)):
                # Make the robot follow the path:
                print('Now path:' + str(i))
                subPath = path[i * 6:(i + 1) * 6]
                res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI', vrep.sim_scripttype_childscript, 'runThroughPath', [self.robot_handle], subPath, [], self.emptyBuff, vrep.simx_opmode_oneshot_wait)
                # Wait until the end of the movement:
                runningPath = True
                while runningPath:
                    res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI', vrep.sim_scripttype_childscript, 'isRunningThroughPath', [self.robot_handle], [], [], self.emptyBuff, vrep.simx_opmode_oneshot_wait)
                    runningPath = retInts[0] == 1
                #time.sleep(1)

    def run_through_path(self, path):
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',
                                                                                     vrep.sim_scripttype_childscript,
                                                                                     'runThroughPath',
                                                                                     [self.robot_handle], path, [],
                                                                                     self.emptyBuff,
                                                                                     vrep.simx_opmode_oneshot_wait)
        # Wait until the end of the movement:
        runningPath = True
        while runningPath:
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',
                                                                                         vrep.sim_scripttype_childscript,
                                                                                         'isRunningThroughPath',
                                                                                         [self.robot_handle], [], [],
                                                                                         self.emptyBuff,
                                                                                         vrep.simx_opmode_oneshot_wait)
            runningPath = retInts[0] == 1

    def visualize_path(self, path):
        # Visualize the path:
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',
                                                                                     vrep.sim_scripttype_childscript,
                                                                                     'visualizePath',
                                                                                     [self.robot_handle, 255, 0, 255],
                                                                                     path, [], self.emptyBuff,
                                                                                     vrep.simx_opmode_oneshot_wait)
        lineHandle = retInts[0]
        return lineHandle

    def clear_path_visualization(self, lineHandle):
        # Clear the path visualization:
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',
                                                                                     vrep.sim_scripttype_childscript,
                                                                                     'removeLine', [lineHandle], [], [],
                                                                                     self.emptyBuff,
                                                                                     vrep.simx_opmode_oneshot_wait)
    def enableDynamic(self, obj_name, enable):
        sim_ret, object_handle = vrep.simxGetObjectHandle(self.clientID, obj_name, vrep.simx_opmode_oneshot_wait)
        enable = int(enable)
        inInts = [object_handle, enable]
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',
                                                                                   vrep.sim_scripttype_childscript,
                                                                                   'enableDynamic', inInts, [],
                                                                                   [], self.emptyBuff,
                                                                                   vrep.simx_opmode_oneshot_wait)
    def setObjectParent(self, obj_name, obj_parent_name):
        sim_ret, object_handle = vrep.simxGetObjectHandle(self.clientID, obj_name, vrep.simx_opmode_oneshot_wait)
        sim_ret, object_parent_handle = vrep.simxGetObjectHandle(self.clientID, obj_parent_name, vrep.simx_opmode_oneshot_wait)
        inInts = [object_handle, object_parent_handle]
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, 'RemoteAPI',
                                                                                     vrep.sim_scripttype_childscript,
                                                                                     'setObjectParent', inInts, [],
                                                                                     [], self.emptyBuff,
                                                                                     vrep.simx_opmode_oneshot_wait)

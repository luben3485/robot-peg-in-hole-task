import cv2
import numpy as np
import random
import torch.utils.data as data
import mankey.config.parameter as parameter
import sys
sys.path.append('/tmp2/r09944001/robot-peg-in-hole-task')
from mankey.utils.imgproc import PixelCoord, get_guassian_heatmap, get_bbox_cropped_image_path
from mankey.dataproc.supervised_keypoint_pcd_db import SupervisedKeypointDBEntry, SupervisedImageKeypointDatabase
import attr


@attr.s
class SupervisedKeypointDatasetConfig:
    # General configuration
    image_database_list = []
    network_in_patch_width = 0
    network_in_patch_height = 0
    network_out_map_width = 0
    network_out_map_height = 0
    is_train = True

    # The bounding box given by Database is tight, make it losser
    bbox_scale = parameter.bbox_scale

    # The normalization parameter
    depth_image_clip = parameter.depth_image_clip  # Clip the depth image further than 1500 mm
    depth_image_mean = parameter.depth_image_mean
    depth_image_scale = parameter.depth_image_scale  # scaled_depth = (raw_depth - depth_image_mean) / depth_image_scale

    # For RGB value
    rgb_mean = parameter.rgb_mean

    # The augmentation parameter
    aug_scale_factor = 0.1
    aug_rot_rad_factor = 10.0 * np.pi / 180.0
    aug_rot_rate = 0.6  # The probability of do rotation augmentation
    aug_color_factor = 0.2

    # Check whether the value is valid
    def sanity_check(self) -> bool:
        if self.network_in_patch_height <= 0 or self.network_in_patch_width <= 0:
            return False

        if self.network_out_map_height <= 0 or self.network_out_map_width <= 0:
            return False

        # OK
        return True


@attr.s
class ProcessedEntry:
    # (height, width, 3) rgb image cropped by bounding box, the pixel value in [0, 255]
    cropped_rgb = np.ndarray(shape=[])

    # The image transfrom from tight bounding box to cropped patch
    bbox2patch = np.ndarray(shape=[])

    # The np array in (n_keypoint, 3). The pixel xy are in cropped_rgb.shape, the depth is in mm.
    keypoint_xy_depth = np.ndarray(shape=[])
    keypoint_validity = np.ndarray(shape=[])

    # The target heatmap value
    target_heatmap = np.ndarray(shape=[])

    # The optional info
    cropped_depth = np.ndarray(shape=[])  # (width, height) depth image, the depth is in mm
    cropped_binary_mask = np.ndarray(shape=[])  # (width, height) 0-1 mask

    # xyzrot
    delta_rotation_matrix = np.ndarray(shape=[])
    delta_translation = np.ndarray(shape=[])
    unit_delta_translation = np.ndarray(shape=[])
    gripper_pose = np.ndarray(shape=[])
    step_size = np.ndarray(shape=[])

    # pcd
    pcd = np.ndarray(shape=[])
    heatmap = np.ndarray(shape=[])
    segmentation = np.ndarray(shape=[])
    kpt_of = np.ndarray(shape=[])
    
    # Some method to check the existance of entry
    @property
    def has_depth(self):
        return True
        #return self.cropped_depth.shape == self.cropped_rgb.shape[0:2]

    @property
    def has_mask(self):
        return self.cropped_binary_mask.shape == self.cropped_rgb.shape[0:2]
    

class SupervisedKeypointDataset(data.Dataset):

    def __init__(self, config: SupervisedKeypointDatasetConfig):
        # General config
        assert config.sanity_check()
        self._config = config
        self._network_in_patch_width = config.network_in_patch_width
        self._network_in_patch_height = config.network_in_patch_height
        self._network_out_map_width = config.network_out_map_width
        self._network_out_map_height = config.network_out_map_height
        assert self._network_in_patch_height == self._network_in_patch_width  # Currently only support unit aspect ratio
        self._is_train = config.is_train

        # Build the flatten list of entry
        self._entry_list = []
        self._num_keypoints = -1
        if len(config.image_database_list) == 1:
            self._entry_list = config.image_database_list[0].get_entry_list()
            self._num_keypoints = config.image_database_list[0].num_keypoints
        elif len(config.image_database_list) > 1:
            for database in config.image_database_list:
                # For the size
                if self._num_keypoints < 0:
                    self._num_keypoints = database.num_keypoints
                else:
                    assert self._num_keypoints == database.num_keypoints

                # For all the entry
                for entry in database.get_entry_list():
                    self._entry_list.append(entry)
        else:
            raise RuntimeError('No database is provided. Exit!')

    def __getitem__(self, index):
        processed_entry = self.get_processed_entry(self._entry_list[index])
        # Do normalization on images
        from mankey.utils.imgproc import rgb_image_normalize, depth_image_normalize
        # The randomization on rgb
        color_aug_scale = self._get_color_randomization_parameter()
        normalized_rgb = []
        for cropped_rgb in processed_entry.cropped_rgb:   
            rgb = rgb_image_normalize(cropped_rgb, self._config.rgb_mean, color_aug_scale)
            normalized_rgb.append(rgb)
        rgb_channels, height, width = normalized_rgb[0].shape
        
        # Check the total size of tensor
        tensor_channels = rgb_channels
        if processed_entry.has_depth:
            tensor_channels += 1

        # Construct the tensor
        stacked_tensor = np.zeros(shape=(tensor_channels, height, width), dtype=np.float32)
        stacked_tensor[0:rgb_channels, :, :] = normalized_rgb[0]

        # Process other channels
        channel_offset = rgb_channels
        if processed_entry.has_depth:
            # The depth should not be randomized
            normalized_depth = []
            for cropped_depth in processed_entry.cropped_depth:
                cropped_depth[cropped_depth >= self._config.depth_image_clip] = 0
                depth = cv2.normalize(cropped_depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                normalized_depth.append(depth)
            ''' 
            normalized_depth = depth_image_normalize(
                processed_entry.cropped_depth,
                self._config.depth_image_clip,
                self._config.depth_image_mean,
                self._config.depth_image_scale)
            '''
            stacked_tensor[channel_offset, :, :] = normalized_depth[0]
            channel_offset += 1
            
        #rgb_pair = np.concatenate((normalized_rgb[0],normalized_rgb[1]), axis=0)
        #depth_pair = np.stack(normalized_depth)

        # Do scale on keypoint xy and depth
        normalized_keypoint_xy_depth = processed_entry.keypoint_xy_depth.copy()
        normalized_keypoint_xy_depth[0, :] = (processed_entry.keypoint_xy_depth[0, :] / float(width)) -0.5
        normalized_keypoint_xy_depth[1, :] = (processed_entry.keypoint_xy_depth[1, :] / float(height)) -0.5
        normalized_keypoint_xy_depth[2, :] = \
            (processed_entry.keypoint_xy_depth[2, :] - self._config.depth_image_mean) / float(self._config.depth_image_scale)
        normalized_keypoint_xy_depth = np.transpose(normalized_keypoint_xy_depth, (1, 0))

        # OK
        validity = np.transpose(processed_entry.keypoint_validity, (1, 0))
        return {
            parameter.rgbd_image_key: stacked_tensor,
            #parameter.rgb_pair_key: rgb_pair,
            #parameter.depth_pair_key: depth_pair,
            parameter.pcd_key: processed_entry.pcd.astype(np.float32),
            parameter.heatmap_key: processed_entry.heatmap.astype(np.float32),
            parameter.segmentation_key: processed_entry.segmentation.astype(np.float32),
            parameter.kpt_of_key: processed_entry.kpt_of.astype(np.float32),
            parameter.keypoint_xyd_key: normalized_keypoint_xy_depth.astype(np.float32),
            parameter.keypoint_validity_key: validity.astype(np.float32),
            parameter.target_heatmap_key: processed_entry.target_heatmap.astype(np.float32),
            parameter.delta_rot_key: processed_entry.delta_rotation_matrix.astype(np.float32),
            parameter.delta_rot_euler_key: processed_entry.delta_rot_euler.astype(np.float32),
            #parameter.delta_rot_cls_key: processed_entry.delta_rot_cls.astype(np.int),
            parameter.delta_xyz_key: processed_entry.delta_translation.astype(np.float32),
            parameter.unit_delta_xyz_key: processed_entry.unit_delta_translation.astype(np.float32),
            parameter.gripper_pose_key: processed_entry.gripper_pose.astype(np.float32),
            parameter.step_size_key: processed_entry.step_size.astype(np.float32),
            parameter.pcd_centroid_key: processed_entry.pcd_centroid.astype(np.float32),
            parameter.pcd_mean_key: processed_entry.pcd_mean.astype(np.float32)
        }
        #return stacked_tensor, normalized_keypoint_xy_depth.astype(np.float32), \
        #       validity.astype(np.float32), processed_entry.target_heatmap.astype(np.float32)

    def __len__(self):
        return len(self._entry_list)

    @property
    def entry_list(self):
        return self._entry_list

    @property
    def num_keypoints(self):
        return self._num_keypoints

    def get_processed_entry(self, entry: SupervisedKeypointDBEntry) -> ProcessedEntry:
        """
        Do image processing given the input entry
        The entry only contains path to image and tight bounding box, this method
        load the image and process them.
        :param entry:
        :return:
        """
        processed_entry = ProcessedEntry()

        # The augmentation parameter
        scale, rot_rad = self._get_geometric_augmentation_parameter(entry)

        # The rgb image
        warped_rgb = []

        for rgb_path in entry.rgb_image_path:
            rgb, bbox2patch = get_bbox_cropped_image_path(
                imgpath=rgb_path, is_rgb=True,
                bbox_topleft=entry.bbox_top_left, bbox_bottomright=entry.bbox_bottom_right,
                patch_width=self._network_in_patch_width, patch_height=self._network_in_patch_height,
                bbox_scale=self._config.bbox_scale, on_boundary=entry.on_boundary,
                scale=scale, rot_rad=rot_rad)
            warped_rgb.append(rgb)
            
        # Transform the keypoint
        pixelxy_depth, validity = self._get_transformed_keypoint(
            bbox2patch, entry,
            self._network_in_patch_width, self._network_in_patch_height)

        # Save to processed_entry, these info must be there
        processed_entry.cropped_rgb = warped_rgb
        processed_entry.bbox2patch = bbox2patch
        processed_entry.keypoint_xy_depth = pixelxy_depth
        processed_entry.keypoint_validity = validity
        
        # xyzrot
        processed_entry.delta_translation = entry.delta_translation
        processed_entry.unit_delta_translation = entry.unit_delta_translation
        processed_entry.delta_rotation_matrix = entry.delta_rotation_matrix
        processed_entry.delta_rot_euler = entry.delta_rot_euler
        processed_entry.gripper_pose = entry.gripper_pose
        processed_entry.step_size = entry.step_size
        #processed_entry.delta_rot_cls = entry.delta_rot_cls
        
        # pcd
        pcd_raw = np.load(entry.pcd_path)
        processed_entry.pcd = pcd_raw[:, :3].reshape(-1,3)
        processed_entry.heatmap = pcd_raw[:, 4].reshape(-1,1)
        processed_entry.segmentation = pcd_raw[:, 3].reshape(-1,1)
        processed_entry.kpt_of = pcd_raw[:, 5:].reshape(-1,9)
        processed_entry.pcd_centroid = entry.pcd_centroid.reshape(3,)
        processed_entry.pcd_mean = entry.pcd_mean.reshape(1,)
                              
        # Compute the guassian heatmap
        n_keypoint = pixelxy_depth.shape[1]
        processed_entry.target_heatmap = np.zeros(shape=(
            n_keypoint,
            self._network_out_map_height,
            self._network_out_map_width))

        # Iterate over keypoints: note the scale different on keypoints
        ratio = self._network_out_map_width / self._network_in_patch_width
        for i in range(n_keypoint):
            processed_entry.target_heatmap[i, :, :] = get_guassian_heatmap(
                pixelxy_depth[0:2, i] * ratio,
                heatmap_size=self._network_out_map_width)

        # The depth image
        if entry.has_depth:
            warped_depth = []
            for depth_path in entry.depth_image_path:
                depth, _ = get_bbox_cropped_image_path(
                    imgpath=depth_path, is_rgb=False,
                    bbox_topleft=entry.bbox_top_left, bbox_bottomright=entry.bbox_bottom_right,
                    patch_width=self._network_in_patch_width, patch_height=self._network_in_patch_height,
                    bbox_scale=self._config.bbox_scale, on_boundary=entry.on_boundary,
                    scale=scale, rot_rad=rot_rad)
                warped_depth.append(depth)
            processed_entry.cropped_depth = warped_depth

        # The binary mask
        if entry.has_mask:
            warped_mask, _ = get_bbox_cropped_image_path(
                imgpath=entry.binary_mask_path, is_rgb=False,
                bbox_topleft=entry.bbox_top_left, bbox_bottomright=entry.bbox_bottom_right,
                patch_width=self._network_in_patch_width, patch_height=self._network_in_patch_height,
                bbox_scale=self._config.bbox_scale, on_boundary=entry.on_boundary,
                scale=scale, rot_rad=rot_rad)
            processed_entry.cropped_binary_mask = warped_mask

        # Seems ok
        return processed_entry

    def _get_geometric_augmentation_parameter(self, entry: SupervisedKeypointDBEntry) -> (float, float):
        """
        From the config and entry, get the parameter used for augmentation
        :param entry:
        :return: scale, rotation
        """
        # Not training
        if not self._is_train:
            return 1.0, 0.0

        # For scale
        scale = np.clip(np.random.randn(), -1.0, 1.0) * self._config.aug_scale_factor + 1.0

        # For rotate:
        if random.random() < self._config.aug_rot_rate and (not entry.on_boundary):
            rotate_rad = np.clip(np.random.randn(), -2.0, 2.0) * self._config.aug_rot_rad_factor
        else:
            rotate_rad = 0.0

        # OK
        return scale, rotate_rad

    def _get_color_randomization_parameter(self):
        # Not training
        if not self._is_train:
            return [1.0, 1.0, 1.0]

        # Need augmentation here
        c_up = 1.0 + self._config.aug_color_factor
        c_low = 1.0 - self._config.aug_color_factor
        color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
        return color_scale

    @staticmethod
    def _get_transformed_keypoint(
            transform: np.ndarray,
            entry: SupervisedKeypointDBEntry,
            patch_width: int, patch_height: int) -> (np.ndarray, np.ndarray):
        """
        Given the bounding box to patch transform, compute the transform keypoint
        and their validity. Note that transformed pixel might not be int
        :param transform: 3x3 homogeneous transform matrix
        :param entry:
        :param patch_width:
        :param patch_height:
        :return: A tuple contains the transformed pixelxy_depth and validity
        """
        from mankey.utils.imgproc import transform_2d, PixelCoord, pixel_in_bbox

        # Allocate the space
        n_keypoint = entry.keypoint_pixelxy_depth.shape[1]
        transformed_pixelxy_depth = np.zeros((3, n_keypoint))
        transformed_validity_weight = np.ones((3, n_keypoint))

        # Construct bounding box
        top_left = PixelCoord()
        top_left.x = 0
        top_left.y = 0
        bottom_right = PixelCoord()
        bottom_right.x = patch_width
        bottom_right.y = patch_height

        # Do transform
        pixel = PixelCoord()
        for i in range(n_keypoint):
            transformed_pixelxy_depth[0:2, i] = transform_2d(entry.keypoint_pixelxy_depth[0:2, i], transform)
            transformed_pixelxy_depth[2, i] = entry.keypoint_pixelxy_depth[2, i]

            # Check validity
            pixel.x = int(transformed_pixelxy_depth[0, i])
            pixel.y = int(transformed_pixelxy_depth[1, i])
            if not pixel_in_bbox(pixel, top_left, bottom_right):
                transformed_validity_weight[0, i] = 0
                transformed_validity_weight[1, i] = 0
                transformed_validity_weight[2, i] = 0

        # OK
        return transformed_pixelxy_depth, transformed_validity_weight


# Simple test code
def save_loaded_img():
    from mankey.dataproc.spartan_supervised_db import SpartanSupvervisedKeypointDBConfig, SpartanSupervisedKeypointDatabase

    # Construct the db
    db_config = SpartanSupvervisedKeypointDBConfig()
    db_config.keypoint_yaml_name = 'shoe_6_keypoint_image.yaml'
    db_config.pdc_data_root = '/home/wei/data/pdc'
    db_config.config_file_path = '/home/wei/Coding/mankey/config/boot_logs.txt'
    database = SpartanSupervisedKeypointDatabase(db_config)

    # Construct torch dataset
    config = SupervisedKeypointDatasetConfig()
    config.network_in_patch_width = 256
    config.network_in_patch_height = 256
    config.network_out_map_width = 64
    config.network_out_map_height = 64
    config.image_database_list.append(database)
    config.is_train = False
    dataset = SupervisedKeypointDataset(config)

    # Simple check
    import os
    print(len(dataset))
    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # Save all the warped image
    from mankey.utils.imgproc import draw_image_keypoint, draw_visible_heatmap, get_visible_mask
    for i in range(min(1000, len(dataset))):
        idx = random.randint(0, len(dataset) - 1)
        processed_entry = dataset.get_processed_entry(dataset.entry_list[idx])
        rgb_keypoint = draw_image_keypoint(processed_entry.cropped_rgb, processed_entry.keypoint_xy_depth, processed_entry.keypoint_validity)
        cv2.imwrite(os.path.join(tmp_dir, 'image_%d_rgb.png' % idx), rgb_keypoint)
        cv2.imwrite(os.path.join(tmp_dir, 'mask_image_%d_rgb.png' % idx),
                    get_visible_mask(processed_entry.cropped_binary_mask))
        # cv2.imwrite(os.path.join(tmp_dir, 'image_%d_depth.png' % i), get_visible_depth(processed_entry.cropped_depth))
        # cv2.imwrite(os.path.join(tmp_dir, 'image_%d_heatmap.png' % i), heatmap)


if __name__ == '__main__':
    save_loaded_img()

# The string key for dataset
rgbd_image_key = 'rgbd_image'
pcd_key = 'pcd'
heatmap_key = 'heatmap'
segmentation_key = 'segmentation'
keypoint_xyd_key = 'normalized_keypoint_xyd'
keypoint_validity_key = 'validity'
target_heatmap_key = 'target_heatmap'
# xyzrot
delta_rot_key = 'delta_rotation_matrix'
delta_rot_cls_key = 'cls'
delta_xyz_key = 'delta_translation'
unit_delta_xyz_key = 'unit_delta_translation'
gripper_pose_key = 'gripper_pose'
step_size_key = 'step_size'
kpt_of_key = 'kpt_of'
pcd_centroid_key = 'pcd_centroid'
pcd_mean_key = 'pcd_mean_key'

# The bounding box given by Database is tight, make it losser
bbox_scale = 1.25

# The normalization parameter
depth_image_clip = 2000  # Clip the depth image further than 1500 mm
depth_image_mean = 650   # origin 580
depth_image_scale = 256  # scaled_depth = (raw_depth - depth_image_mean) / depth_image_scale

# The averaged RGB image
rgb_mean = [0.485, 0.456, 0.406]

# The default size of path
default_patch_size_input = 256
default_patch_size_output = 64

# The camera parameter
focal_x = 309.019
focal_y = 309.019
principal_x = 128
principal_y = 128

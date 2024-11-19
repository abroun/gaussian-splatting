import math
import os.path
import sys

import cv2
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData
import numpy as np

from .beholder_project import BeholderProject

def deg2rad(deg):
    return deg*math.pi/180.0

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

IMAGE_WIDTH = 4000
IMAGE_HEIGHT = 3000
FOCAL_LENGTH_X = 946
FOCAL_LENGTH_Y = 946

K_MAX_VAL = 1.0
K_NUM_STEPS = 1000

F_MAX_VAL = 2000
F_NUM_STEPS = 1000

C_NUM_STEPS = 1000
C_OFFSET_X = -29
C_OFFSET_Y = -11

IMAGE_SCALE = 4000/3840
CAMERA_0_FX = 937.444559943624 * IMAGE_SCALE
CAMERA_0_FY = CAMERA_0_FX # 702.8304278608501
CAMERA_0_C_OFFSET_X = (1894.153291774264 - 3840/2)*IMAGE_SCALE
CAMERA_0_C_OFFSET_Y = (1086.4121630293876 - 2160/2)*IMAGE_SCALE
CAMERA_0_DIST = np.array([-0.023588274364117236, -0.002327597684324214, -0.0008498552353308531, -2.644155110680429e-05], dtype=np.float32)

CAMERA_1_FX = 933.727726565967 * IMAGE_SCALE
CAMERA_1_FY = CAMERA_1_FX # 700.0933939950648
CAMERA_1_C_OFFSET_X = (1872.8785779227594 - 3840/2)*IMAGE_SCALE
CAMERA_1_C_OFFSET_Y = (1087.019068424728 - 2160/2)*IMAGE_SCALE
CAMERA_1_DIST = np.array([-0.023474597361757957, -0.0011375684959457504, -0.001493101972596532, 8.421108249985315e-05], dtype=np.float32)

def k2step(k):
    k = max(0, min(k, K_MAX_VAL))
    step = int(k/K_MAX_VAL * K_NUM_STEPS)

    return step

def step2k(step):
    step = max(0, min(step, K_NUM_STEPS))
    k = step/K_NUM_STEPS * K_MAX_VAL

    return k

def f2step(f):
    f = max(0, min(f, F_MAX_VAL))
    step = int(f/F_MAX_VAL * F_NUM_STEPS)

    return step

def step2f(step):
    step = max(0, min(step, F_NUM_STEPS))
    f = step/F_NUM_STEPS * F_MAX_VAL

    return f

def c2step(c):
    min_c = -C_NUM_STEPS/2
    max_c = C_NUM_STEPS/2

    c = max(min_c, min(c, max_c))
    step = int(c + C_NUM_STEPS/2)

    return step

def step2c(step):
    step = max(0, min(step, C_NUM_STEPS))
    c = step - C_NUM_STEPS/2

    return c

def render_point_cloud_quaternion(point_cloud_positions, point_cloud_rgb, proj_matrix, cam_R, cam_T, image_size=(800, 800)):
    """
    Render a point cloud from the perspective of a pinhole camera using OpenCV, with rotation provided as a quaternion.

    Parameters:
    - point_cloud_positions (numpy array): Nx3 array of 3D points in world coordinates.
    - point_cloud_rgb (numpy array): Nx3 array of RGB point colours
    - proj_matrix (numpy array): 3x3 intrinsic matrix of the pinhole camera.
    - quaternion (numpy array): 4x1 quaternion (x, y, z, w) for the camera rotation.
    - tvec (numpy array): 3x1 translation vector for the camera pose.
    - image_size (tuple): Size of the output image (width, height).

    Returns:
    - image (numpy array): Rendered image of the point cloud.
    """
    # Convert point cloud to homogeneous coordinates
    points_homogeneous = np.hstack((point_cloud_positions, np.ones((point_cloud_positions.shape[0], 1))))
    #print(points_homogeneous)
    
    # Convert quaternion to rotation matrix
    rot_mtx = cam_R

    inv_rot_mtx = rot_mtx.T
    inv_t = -inv_rot_mtx @ cam_T
    
    # Build transformation matrix using the rotation matrix and translation vector
    transformation_matrix = np.hstack((inv_rot_mtx, inv_t.reshape(-1, 1)))

    print("K")
    print(proj_matrix)
    print("transformation_matrix")
    print(transformation_matrix)

    points_camera = (transformation_matrix @ points_homogeneous.T).T

    cv2_points_camera = points_camera[:, :3].astype(np.float32)
    cv2_points_camera = (proj_matrix @ cv2_points_camera.T).T

    # Create a blank image
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # Scale and draw points on the image
    num_points = cv2_points_camera.shape[0]
    num_points_culled = 0
    for point_idx in range(num_points):

        camera_point = cv2_points_camera[point_idx, :]

        # Filter out points behind the camera
        if camera_point[2] <= 0:
            num_points_culled += 1
            continue

        x, y = int(camera_point[0]/camera_point[2]), int(camera_point[1]/camera_point[2])

        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:  # Ensure points are within image bounds
            image[y, x, 0] = point_cloud_rgb[point_idx, 2]
            image[y, x, 1] = point_cloud_rgb[point_idx, 1]
            image[y, x, 2] = point_cloud_rgb[point_idx, 0]
            #cv2.circle(image, (x, y), 1, (255, 255, 255), -1)  # Draw points as white dots

    return image

def render_point_cloud_fisheye(point_cloud_positions, point_cloud_rgb, proj_matrix, dist_coeffs, cam_R, cam_T, image_size=(800, 800)):
    """
    Render a point cloud from the perspective of a fisheye camera using OpenCV, with rotation provided as a quaternion.

    Parameters:
    - point_cloud_positions (numpy array): Nx3 array of 3D points in world coordinates.
    - point_cloud_rgb (numpy array): Nx3 array of RGB point colours
    - proj_matrix (numpy array): 3x3 intrinsic matrix of the fisheye camera.
    - dist_coeffs (numpy array): 4x1 distortion coefficients for the fisheye camera.
    - quaternion (numpy array): 4x1 quaternion (x, y, z, w) for the camera rotation.
    - tvec (numpy array): 3x1 translation vector for the camera pose.
    - image_size (tuple): Size of the output image (width, height).

    Returns:
    - image (numpy array): Rendered image of the point cloud.
    """
    # Convert point cloud to homogeneous coordinates
    points_homogeneous = np.hstack((point_cloud_positions, np.ones((point_cloud_positions.shape[0], 1))))
    #print(points_homogeneous)
    
    # Convert quaternion to rotation matrix
    rot_mtx = cam_R

    print("rot_mtx")
    print(rot_mtx)

    inv_rot_mtx = rot_mtx.T
    inv_t = -inv_rot_mtx @ cam_T
    
    print("inv_rot_mtx")
    print(inv_rot_mtx)
    print("inv_t")
    print(inv_t)

    # Build transformation matrix using the rotation matrix and translation vector
    transformation_matrix = np.hstack((inv_rot_mtx, inv_t.reshape(-1, 1)))

    print(transformation_matrix.shape)

    points_camera = (transformation_matrix @ points_homogeneous.T).T

    cv2_points_camera = points_camera[:, :3].astype(np.float32)
    cv2_points_camera = np.expand_dims(cv2_points_camera, axis=0)

    # print(cv2_points_camera.shape)
    # cv2_points_camera = np.zeros((1, 30, 3), dtype=np.float32)
    # print(cv2_points_camera.shape)

    

    # Project 3D points to 2D fisheye image plane
    print("K")
    print(proj_matrix.shape, proj_matrix.dtype)
    print("D")
    print(dist_coeffs.shape, dist_coeffs.dtype)

    image_points, _ = cv2.fisheye.projectPoints(cv2_points_camera, np.zeros(3), np.zeros(3), proj_matrix, dist_coeffs, alpha=0) #math.pi/2)

    print(image_points.shape)

    # Create a blank image
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # Scale and draw points on the image
    num_points = image_points.shape[1]
    num_points_culled = 0
    for point_idx in range(num_points):

        point = image_points[0, point_idx, :]
        x, y = int(point[0]), int(point[1])

        # Filter out points behind the camera
        if points_camera[point_idx, 2] <= 0:
            num_points_culled += 1
            continue

        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:  # Ensure points are within image bounds
            image[y, x, 0] = point_cloud_rgb[point_idx, 2]
            image[y, x, 1] = point_cloud_rgb[point_idx, 1]
            image[y, x, 2] = point_cloud_rgb[point_idx, 0]
            #cv2.circle(image, (x, y), 1, (255, 255, 255), -1)  # Draw points as white dots

    print(f"Culled {num_points_culled} points")

    return image

def rectify_fisheye_image(fisheye_image, K, D, image_size, balance=0.0, fov_scale=1.0):
    """
    Rectify a fisheye image to a pinhole camera model.

    Parameters:
    - fisheye_image (numpy array): Input fisheye image.
    - K (numpy array): 3x3 camera intrinsic matrix.
    - D (numpy array): 4x1 distortion coefficients for the fisheye camera.
    - image_size (tuple): Size of the output rectified image (width, height).
    - balance (float): Balance between preserving edges and straightening lines (0 to 1).
    - fov_scale (float): Scaling factor for the field of view in the rectified image.

    Returns:
    - rectified_image (numpy array): Output rectified image.
    """
    # Compute the new optimal intrinsic matrix for the rectified image
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, image_size, np.eye(3), balance=balance, new_size=image_size, fov_scale=fov_scale)

    # Initialize the undistortion and rectification map
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, image_size, cv2.CV_16SC2)

    # Apply the remapping to obtain the rectified image
    rectified_image = cv2.remap(fisheye_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return rectified_image, K_new


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} beholder_filename cam_idx")
        sys.exit(-1)

    beholder_filename = sys.argv[1]
    cam_idx = int(sys.argv[2])
    
    camera_found = False
    project_dirname = os.path.dirname(beholder_filename)

    beholder_project = BeholderProject.load_from_file(beholder_filename)
    image_info = beholder_project.get_camera_image_info(cam_idx)

    image_path = image_info["path"]
    camera_name = image_info["name"]

    cam_R, cam_T = beholder_project.get_camera_extrinsics(cam_idx)
    width, height, K, dist_coeffs, cam_model = beholder_project.get_camera_intrinsics_and_dist_coeffs(cam_idx)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Read in the point cloud
    point_cloud_data = beholder_project.load_point_cloud()
    point_cloud_positions = point_cloud_data[:, :3]
    point_cloud_rgb = (point_cloud_data[:, 3:]*255.0).astype(np.uint8)

    # Load in the original image
    orig_image = cv2.imread(image_path)
    orig_image_weight = 0.1

    def update_trackbar(*args):
        pass

    def update_render():

        # Setup camera parameters
        print("Focal length X (pixels):", fx)
        print("Focal length X (pixels):", fy)
        print("offset cx:", cx)
        print("offset cy:", cy)
        proj_mtx = np.eye(3, dtype=np.float32)
        proj_mtx[0, 0] = fx
        proj_mtx[1, 1] = fy
        #proj_mtx[0, 2] = IMAGE_WIDTH/2.0 + cx
        #proj_mtx[1, 2] = IMAGE_HEIGHT/2.0 + cy
        proj_mtx[0, 2] = cx
        proj_mtx[1, 2] = cy

        
        print("Dist:", dist_coeffs)

        image = render_point_cloud_fisheye(point_cloud_positions, point_cloud_rgb, proj_mtx, dist_coeffs, cam_R, cam_T, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        image = cv2.addWeighted(orig_image, orig_image_weight, image, 1 - orig_image_weight, 0)
        image = cv2.resize(image, dsize=None, dst=None, fx=0.3, fy=0.3)

        cv2.imshow("Fisheye Image", image)

        rectified_image, K_new = rectify_fisheye_image(orig_image, proj_mtx, dist_coeffs, (IMAGE_WIDTH, IMAGE_HEIGHT), balance=0.0, fov_scale=0.75)
        rectified_point_image = render_point_cloud_quaternion(point_cloud_positions, point_cloud_rgb, K_new, cam_R, cam_T, (IMAGE_WIDTH, IMAGE_HEIGHT))

        rectified_point_image = cv2.addWeighted(rectified_image, orig_image_weight, rectified_point_image, 1 - orig_image_weight, 0)
        rectified_point_image = cv2.resize(rectified_point_image, dsize=None, dst=None, fx=0.3, fy=0.3)

        cv2.imshow("Rectified Image", rectified_point_image)
        

    INIT_K1 = 0
    INIT_K2 = 0
    INIT_K3 = 0
    INIT_K4 = 0

    cv2.namedWindow("Fisheye Image")
    cv2.createTrackbar("f", "Fisheye Image", f2step(FOCAL_LENGTH_X), F_NUM_STEPS, update_trackbar)
    cv2.createTrackbar("cx", "Fisheye Image", c2step(C_OFFSET_X), C_NUM_STEPS, update_trackbar)
    cv2.createTrackbar("cy", "Fisheye Image", c2step(C_OFFSET_Y), C_NUM_STEPS, update_trackbar)
    cv2.createTrackbar("k1", "Fisheye Image", k2step(INIT_K1), K_NUM_STEPS, update_trackbar)
    cv2.createTrackbar("k2", "Fisheye Image", k2step(INIT_K2), K_NUM_STEPS, update_trackbar)
    cv2.createTrackbar("k3", "Fisheye Image", k2step(INIT_K3), K_NUM_STEPS, update_trackbar)
    cv2.createTrackbar("k4", "Fisheye Image", k2step(INIT_K4), K_NUM_STEPS, update_trackbar)

    update_render()

    while True:
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            break

        update_render()

    cv2.destroyAllWindows()


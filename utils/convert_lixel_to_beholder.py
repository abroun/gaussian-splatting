
import bson
import math
import os.path
import sys
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData
import numpy as np

def deg2rad(deg):
    return deg*math.pi/180.0

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

IMAGE_WIDTH = 4000
IMAGE_HEIGHT = 3000
FOV_Y = deg2rad(160.0)
FOCAL_LENGTH_Y = fov2focal(FOV_Y, IMAGE_HEIGHT)
FOCAL_LENGTH_X = FOCAL_LENGTH_Y

ORIG_IMAGE_WIDTH = 3840
ORIG_IMAGE_HEIGHT = 2160

IMAGE_SCALE = IMAGE_WIDTH/ORIG_IMAGE_WIDTH
CAMERA_0_FX = 937.444559943624 * IMAGE_SCALE
CAMERA_0_FY = CAMERA_0_FX # 702.8304278608501
CAMERA_0_C_OFFSET_X = (1894.153291774264 - ORIG_IMAGE_WIDTH/2)*IMAGE_SCALE
CAMERA_0_C_OFFSET_Y = (1086.4121630293876 - ORIG_IMAGE_HEIGHT/2)*IMAGE_SCALE
CAMERA_0_DIST = [-0.023588274364117236, -0.002327597684324214, -0.0008498552353308531, -2.644155110680429e-05]

CAMERA_1_FX = 933.727726565967 * IMAGE_SCALE
CAMERA_1_FY = CAMERA_1_FX # 700.0933939950648
CAMERA_1_C_OFFSET_X = (1872.8785779227594 - ORIG_IMAGE_WIDTH/2)*IMAGE_SCALE
CAMERA_1_C_OFFSET_Y = (1087.019068424728 - ORIG_IMAGE_HEIGHT/2)*IMAGE_SCALE
CAMERA_1_DIST = [-0.023474597361757957, -0.0011375684959457504, -0.001493101972596532, 8.421108249985315e-05]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} lixel_traj_filename")
        sys.exit(-1)

    project_data = {
        "beholder": {
            "version": "1.0.1"
        },
        "boundingBox": {},
        "inputImages": [],
        "transform": {
            "position": { "x": 0.0, "y": 0.0, "z": 0.0 },
            "rotation": { "x": 0.0, "y": 0.0, "z": 0.0 }
        },
        "intrinsics": [{
            "id": 0,
            "model": "OpenCVFisheye",
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "params": [
                CAMERA_0_FX,
                CAMERA_0_FY,
                IMAGE_WIDTH/2 + CAMERA_0_C_OFFSET_X,
                IMAGE_HEIGHT/2 + CAMERA_0_C_OFFSET_Y,
                CAMERA_0_DIST[0],
                CAMERA_0_DIST[1],
                CAMERA_0_DIST[2],
                CAMERA_0_DIST[3]
            ]
        },{
            "id": 1,
            "model": "OpenCVFisheye",
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "params": [
                CAMERA_1_FX,
                CAMERA_1_FY,
                IMAGE_WIDTH/2 + CAMERA_1_C_OFFSET_X,
                IMAGE_HEIGHT/2 + CAMERA_1_C_OFFSET_Y,
                CAMERA_1_DIST[0],
                CAMERA_1_DIST[1],
                CAMERA_1_DIST[2],
                CAMERA_1_DIST[3]
            ]
        }],
        "cameras": []
    }

    print("Converting project...")

    lixel_traj_filename = sys.argv[1]
    with open(lixel_traj_filename, "r") as lixel_traj_file:

        for line in lixel_traj_file:
            line = line.strip()
            if line.startswith("#"):
                continue

            _, image_path, t_x, t_y, t_z, q_x, q_y, q_z, q_w = line.split(" ")

            image_path = os.path.join("images", image_path)

            image_id = len(project_data["inputImages"])
            image_name = os.path.basename(image_path)
            camera_name = os.path.basename(os.path.dirname(image_path))
            if camera_name == "camera_0":
                intrinsics_id = 0
            elif camera_name == "camera_1":
                intrinsics_id = 1
            else:
                raise Exception(f"Invalid camera name of {camera_name}")
            
            image_name = camera_name + "_" + image_name

            project_data["inputImages"].append({
                "name": image_name,
                "path": image_path
            })

            camera_rot = R.from_quat([q_x, q_y, q_z, q_w])
            rot_mtx = camera_rot.as_matrix()
            rot_mtx[:, 1] = -rot_mtx[:, 1]  # Flip the Y and Z axes
            rot_mtx[:, 2] = -rot_mtx[:, 2]
            camera_rot = R.from_matrix(rot_mtx)
            euler_angles = camera_rot.as_euler("YXZ")

            camera_id = len(project_data["cameras"])
            project_data["cameras"].append({
                "id": camera_id,
                "intrinsicsId": intrinsics_id,
                "imageIds": [image_id],
                "position": { 
                    "x": float(t_x), 
                    "y": float(t_y), 
                    "z": float(t_z)
                },
                "rotation": { 
                    "x": euler_angles[1], 
                    "y": euler_angles[0], 
                    "z": euler_angles[2] 
                },
                "registered": True
            })
    
    output_dirname = os.path.dirname(lixel_traj_filename)
    output_filename = os.path.join(output_dirname, "project.bson")

    with open(output_filename, "wb") as project_file:
        project_file.write(bson.dumps(project_data))
    
    # Now convert the point cloud to .bson
    # print("Converting point cloud...")
    # ply_filename = os.path.join(output_dirname, "point_cloud.ply")

    # ply_data = PlyData.read(ply_filename)
    # vertices = ply_data["vertex"]
    # num_vertices = len(vertices)

    # point_cloud_data = np.empty((num_vertices, 6), dtype=np.float32)
    # point_cloud_data[:, 0] = vertices["x"]
    # point_cloud_data[:, 1] = vertices["y"]
    # point_cloud_data[:, 2] = vertices["z"]
    # point_cloud_data[:, 3] = vertices["red"]/255.0
    # point_cloud_data[:, 4] = vertices["green"]/255.0
    # point_cloud_data[:, 5] = vertices["blue"]/255.0

    # # Extract every Nth item
    # POINT_CLOUD_SAMPLE_STEP = 20
    # point_cloud_data = point_cloud_data[::POINT_CLOUD_SAMPLE_STEP, :]

    # point_cloud_data = {
    #     "points": point_cloud_data.tolist()
    # }

    # print("Converted point cloud data, now writing out...")
    # point_cloud_filename = os.path.join(output_dirname, "point_cloud.bson")
    # with open(point_cloud_filename, "wb") as point_cloud_file:
    #     point_cloud_file.write(bson.dumps(point_cloud_data))

import bson
import os.path
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation as R

class BeholderProject:

    def __init__(self, bhz_filename, data):

        self._data = data
        self._project_dirname = os.path.dirname(bhz_filename)
        self._project_file_basename = os.path.basename(bhz_filename)

    @property
    def inputImages(self):
        return self._data["inputImages"]
    
    @property
    def intrinsics(self):
        return self._data["intrinsics"]
    
    @property
    def cameras(self):
        return self._data["cameras"]

    def get_intrinsics_and_dist_coeffs(self, intrinsics_id):
    
        found_intrinsics = False
        
        for i in self._data["intrinsics"]:
            if i["id"] != intrinsics_id:
                continue

            # Extract parameters from the intrinsics dict
            width = i["width"]
            height = i["height"]
            model = i["model"]

            if model == "SimplePinhole":
                focal_length_x = i["params"][0]
                focal_length_y = focal_length_x
                principal_x = i["params"][1]
                principal_y = i["params"][1]
                distortion_coeffs = np.array([])
            elif model == "Pinhole":
                focal_length_x = i["params"][0]
                focal_length_y = focal_length_x
                principal_x = i["params"][1]
                principal_y = i["params"][2]
                distortion_coeffs = np.array([])
            elif model == "SimpleRadial":
                focal_length_x = i["params"][0]
                focal_length_y = focal_length_x
                principal_x = i["params"][1]
                principal_y = i["params"][2]
                distortion_coeffs = np.array([i["params"][3], 0, 0, 0])
            elif model == "OpenCVFisheye":
                focal_length_x = i["params"][0]
                focal_length_y = i["params"][1]
                principal_x = i["params"][2]
                principal_y = i["params"][3]
                distortion_coeffs = np.array([i["params"][4], i["params"][5], i["params"][6], i["params"][7]])
            else:
                raise Exception(f"Unhandled camera model of type {model} encountered")

            # Construct the intrinsic matrix
            K = np.array([
                [focal_length_x, 0, principal_x],
                [0, focal_length_y, principal_y],
                [0, 0, 1]
            ], dtype=np.float32)

            distortion_coeffs = distortion_coeffs.astype(np.float32)

            found_intrinsics = True
            break

        if not found_intrinsics:
            raise Exception(f"Unable to find intrinsics {intrinsics_id}")

        return width, height, K, distortion_coeffs, model

    def get_camera_intrinsics_and_dist_coeffs(self, camera_id):

        found_camera = False
        
        for c in self._data["cameras"]:
            if c["id"] != camera_id:
                continue

            intrinsics = self.get_intrinsics_and_dist_coeffs(c["intrinsicsId"])
            
            found_camera = True
            break

        if not found_camera:
            raise Exception(f"Unable to find camera {camera_id}")

        return intrinsics
    
    def get_camera_extrinsics(self, camera_id):
        """
        Returns the rotation matrix and translation vector for a given camera.

        Args:
            camera_id (int): The id of the camera to get extrinsics for

        Returns:
            R_matrix (np.array): Rotation matrix (3x3).
            T_vector (np.array): Translation vector (3x1).
        """

        found_camera = False
        
        for c in self._data["cameras"]:
            if c["id"] != camera_id:
                continue

            # Extract position (translation vector)
            T_vector = np.array([
                c["position"]["x"],
                c["position"]["y"],
                c["position"]["z"]
            ])

            # Extract Euler angles in YXZ order
            euler_angles = [
                c["rotation"]["y"],  # Y rotation
                c["rotation"]["x"],  # X rotation
                c["rotation"]["z"]   # Z rotation
            ]

            # Convert YXZ Euler angles to rotation matrix (NOTE: These are intrinsic angles, i.e. relative to a frame of reference that rotates with the object,
            # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html#scipy.spatial.transform.Rotation.from_euler)
            rotation = R.from_euler("YXZ", euler_angles)
            R_matrix = rotation.as_matrix()

            # print("Read in camera", camera_id)
            # print("Rot", c["rotation"]["x"], c["rotation"]["y"], c["rotation"]["z"])
            # print(R_matrix)

            # print(T_vector)

            # Flip the Y and Z axes
            R_matrix[:, 1] = -R_matrix[:, 1]
            R_matrix[:, 2] = -R_matrix[:, 2]
            
            found_camera = True
            break

        if not found_camera:
            raise Exception(f"Unable to find camera {camera_id}")

        return R_matrix, T_vector
    
    def load_camera_image(self, camera_id, layer_idx=0):

        found_camera = False
        
        for c in self._data["cameras"]:
            if c["id"] != camera_id:
                continue

            image_idx = c["imageIds"][layer_idx]
            image = self.load_image(image_idx)

            found_camera = True
            break

        if not found_camera:
            raise Exception(f"Unable to find camera {camera_id}")

        return image
    
    def get_camera_image_info(self, camera_id, layer_idx=0):

        found_camera = False
        
        for c in self._data["cameras"]:
            if c["id"] != camera_id:
                continue

            image_idx = c["imageIds"][layer_idx]
            image_info = self._data["inputImages"][image_idx]
            if not os.path.isabs(image_info["path"]):
                image_info["path"] = os.path.join(self._project_dirname, image_info["path"])

            found_camera = True
            break

        if not found_camera:
            raise Exception(f"Unable to find camera {camera_id}")

        return image_info
    
    def load_point_cloud(self):
        if self._project_file_basename.lower().endswith(".bhz"):
            # Load from project file
            bhz_filename = os.path.join(self._project_dirname, self._project_file_basename)

            with zipfile.ZipFile(bhz_filename, "r") as zip_file:
                with zip_file.open("point_cloud.bson") as point_cloud_file:
                    bson_data = point_cloud_file.read()
                    point_cloud_data = bson.loads(bson_data)
        else:
            # Load from unpacked project directory
            pointcloud_filename = os.path.join(self._project_dirname, "point_cloud.bson")
            with open(pointcloud_filename, "rb") as point_cloud_file:
                bson_data = point_cloud_file.read()
                point_cloud_data = bson.loads(bson_data)

        return np.array(point_cloud_data["points"], dtype=np.float32)

    @staticmethod
    def load_from_file(bhz_filename):

        # Open the zip file and read project.bson
        with zipfile.ZipFile(bhz_filename, "r") as zip_file:
            with zip_file.open("project.bson") as project_file:
                # Read and decode the BSON file content
                bson_data = project_file.read()
                # Parse the BSON data into a Python dictionary
                data = bson.loads(bson_data)

        return BeholderProject(bhz_filename, data)
    
    @staticmethod
    def load_from_directory(bhz_dirname):
        """Use this if working with an unpacked Beholder project"""

        project_filename = os.path.join(bhz_dirname, "project.bson")
        with open(project_filename, "rb") as project_file:
            # Read and decode the BSON file content
            bson_data = project_file.read()
            # Parse the BSON data into a Python dictionary
            data = bson.loads(bson_data)

        return BeholderProject(project_filename, data)

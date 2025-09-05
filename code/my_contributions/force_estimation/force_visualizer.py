import argparse
import json
from pathlib import Path

import numpy as np
import open3d
from open3d import *
from math import sqrt
import cv2

# ROS is optional. Import only if available / used.
try:  # lazy ROS import to allow non-ROS usage
    import rospy  # type: ignore
    from geometry_msgs.msg import WrenchStamped  # type: ignore
    _ROS_AVAILABLE = True
except Exception:  # ROS not installed
    _ROS_AVAILABLE = False

final_force = False
reference_frame = False


def get_rotation_matrix_from_2_vectors(v2, v1=None):
    if v1 is None:
        v1 = [0, 0, -1]
    v2 = v2 / np.linalg.norm(v2)
    v1 = v1 / np.linalg.norm(v1)
    if np.allclose(v1, v2):
        return np.eye(3)
    if np.allclose(v1, -v2):
        return -np.eye(3)

    origin = v1
    target = np.array([-1, -1, 1]) * v2
    target[0], target[1] = target[1], target[0]
    v = np.cross(origin, target)
    c = np.dot(origin, target)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def draw_vector(x, y, z, site, cylinder_radius=0.5, cone_radius=0.7, cone_height=1):
    x_axis = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius, cone_radius=cone_radius, cylinder_height=abs(x), cone_height=cone_height)
    y_axis = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius, cone_radius=cone_radius, cylinder_height=abs(y), cone_height=cone_height)
    z_axis = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius, cone_radius=cone_radius, cylinder_height=abs(z), cone_height=cone_height)
    xyz = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius, cone_radius=cone_radius, cylinder_height=sqrt(x ** 2 + y ** 2 + z ** 2),
        cone_height=cone_height)

    if x > 0:
        x_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [0, np.pi / 2, 0]), [0, 0, 0])
    else:
        x_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [0, -np.pi / 2, 0]), [0, 0, 0])

    if y > 0:
        y_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [np.pi, 0, 0]), [0, 0, 0])
    # if y > 0:
    #     y_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
    #         [np.pi / 0.85, 0.45, 0.45]), [0, 0, 0])
    # else:
    #     y_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
    #         [np.pi / 2, 0, 0]), [0, 0, 0])

    if z > 0:
        z_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [np.pi / 2, 0, 0]), [0, 0, 0])
    else:
        z_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi / 2, 0, 0]), [0, 0, 0])

    xyz.rotate(get_rotation_matrix_from_2_vectors(
        np.array([1, 1, 1])), [0, 0, 0])

    x_axis.paint_uniform_color([1, 0, 0])
    y_axis.paint_uniform_color([0, 1, 0])
    z_axis.paint_uniform_color([0, 0, 1])
    xyz.paint_uniform_color([0, 0, 0])

    # axis = x_axis + y_axis + z_axis
    x_axis.translate(site)
    y_axis.translate(site)
    z_axis.translate(site)
    xyz.translate(site)

    return x_axis, y_axis, z_axis, xyz


def draw_coordinate_frame(x, y, z, site=None, cylinder_radius=0.05, cone_radius=0.01, cone_height=1):
    if site is None:
        site = [0, 0, 0]
    cx, cy, cz, none = draw_vector(
        x, y, z, site, cylinder_radius, cone_radius, cone_height)
    return cx + cy + cz


class Visualizer:
    def __init__(self):
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name='6D_Force_Estimation', width=2000, height=1200)

        self.force_origin = [-15, 0, 0]
        self.torque_origin = [15, 0, 0]

        if reference_frame:
            self.force_axis = draw_coordinate_frame(
                10, 10, 10, self.force_origin)
            self.force_axis_1 = draw_coordinate_frame(
                -10, -10, -10, self.force_origin)
            self.torque_axis = draw_coordinate_frame(
                10, 10, 10, self.torque_origin)
            self.torque_axis_1 = draw_coordinate_frame(
                -10, -10, -10, self.torque_origin)
            self.vis.add_geometry(self.force_axis)
            self.vis.add_geometry(self.force_axis_1)
            self.vis.add_geometry(self.torque_axis)
            self.vis.add_geometry(self.torque_axis_1)

        self.force_x, self.force_y, self.force_z, self.force_xyz = draw_vector(
            10, 10, 10, self.force_origin)
        self.torque_x, self.torque_y, self.torque_z, self.torque_xyz = draw_vector(
            10, 10, 10, self.torque_origin)

        self.last_wrench = np.ones(6)
        self.last_force = np.ones(3)
        self.last_torque = np.ones(3)

        self.vis.add_geometry(self.force_x)
        self.vis.add_geometry(self.force_y)
        self.vis.add_geometry(self.force_z)
        self.vis.add_geometry(self.torque_x)
        self.vis.add_geometry(self.torque_y)
        self.vis.add_geometry(self.torque_z)

        if final_force:
            self.vis.add_geometry(self.force_xyz)
            self.vis.add_geometry(self.torque_xyz)

        self.ctr = self.vis.get_view_control()
        self.ctr.change_field_of_view(-20)
        # print("fov", self.ctr.get_field_of_view())
        # self.ctr.convert_to_pinhole_camera_parameters()
        self.ctr.set_zoom(0.4)
        self.ctr.rotate(-80, 120)  # mouse drag in x-axis, y-axis
        self.ctr.set_lookat([0, 0, 5])
        self.vis.update_renderer()
        self.wrench = np.array([.5, .5, 1.0, .5, .5, .5])
        self.recall = np.array([-0.5, -0.5, -1, -0.5, -0.5, -0.5])
        self.scale = np.array([8, 8, -4, 10, 10, 20])

    def update_force(self, wrench):
        self.wrench = wrench
        visualized_wrench = (wrench + self.recall) * self.scale
        force = visualized_wrench[:3].copy()
        torque = visualized_wrench[3:].copy()

        last_rotation = get_rotation_matrix_from_2_vectors(
            self.last_force)
        rotation = get_rotation_matrix_from_2_vectors(force)
        rotation = np.matmul(rotation, np.linalg.inv(last_rotation))
        if final_force:
            self.force_xyz.rotate(rotation, center=self.force_origin)
            self.force_xyz.scale(np.linalg.norm(
                force) / np.linalg.norm(self.last_force), center=self.force_origin)
        self.last_force = force

        last_rotation = get_rotation_matrix_from_2_vectors(
            self.last_torque)
        rotation = get_rotation_matrix_from_2_vectors(torque)
        rotation = np.matmul(rotation, np.linalg.inv(last_rotation))
        if final_force:
            self.torque_xyz.rotate(rotation, center=self.torque_origin)
            self.torque_xyz.scale(np.linalg.norm(
                torque) / np.linalg.norm(self.last_torque), center=self.torque_origin)
        self.last_torque = torque

        visualized_wrench[visualized_wrench == 0] = 0.00001
        visualized_wrench /= self.last_wrench
        self.last_wrench = self.last_wrench * visualized_wrench

        self.force_x.scale(visualized_wrench[0], center=self.force_origin)
        self.force_y.scale(visualized_wrench[1], center=self.force_origin)
        self.force_z.scale(visualized_wrench[2], center=self.force_origin)
        self.torque_x.scale(visualized_wrench[3], center=self.torque_origin)
        self.torque_y.scale(visualized_wrench[4], center=self.torque_origin)
        self.torque_z.scale(visualized_wrench[5], center=self.torque_origin)

        self.vis.update_geometry(self.force_x)
        self.vis.update_geometry(self.force_y)
        self.vis.update_geometry(self.force_z)
        self.vis.update_geometry(self.torque_x)
        self.vis.update_geometry(self.torque_y)
        self.vis.update_geometry(self.torque_z)

        if final_force:
            self.vis.update_geometry(self.force_xyz)
            self.vis.update_geometry(self.torque_xyz)

    def ros_callback(self, msg):
        self.wrench = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Force Visualizer (ROS or Static)')
    sub = parser.add_subparsers(dest='mode', required=False)

    # ROS mode (default, requires ROS installed)
    ros_p = sub.add_parser('ros', help='Run ROS visualizer (subscribe to /predicted_wrench)')
    ros_p.add_argument('--node', default='force_visualization', help='ROS node name')

    # Static mode from a single image using the runtime predictor
    img_p = sub.add_parser('image', help='Render a static visualization from an image')
    img_p.add_argument('--cfg', default='shape_reconstruction/shape_config.yaml')
    img_p.add_argument('--sensor_id', type=int, default=3)
    img_p.add_argument('--ref', required=True, help='Reference image (lowest-force)')
    img_p.add_argument('--img', required=True, help='Target image to visualize')
    img_p.add_argument('--coef', default='force_estimation/analysis/force_regression_linear_mean.json',
                      help='Linear regression coefficients JSON (intercept, coef_mean_depth)')
    img_p.add_argument('--out', default='force_estimation/force_visualization.png', help='Output PNG path')

    args = parser.parse_args()

    if args.mode == 'ros' or (args.mode is None and _ROS_AVAILABLE):
        if not _ROS_AVAILABLE:
            raise RuntimeError('ROS mode requested but ROS is not available in this environment.')
        visualizer = Visualizer()
        rospy.init_node(args.node)
        predicted_force_sub = rospy.Subscriber('/predicted_wrench', WrenchStamped, visualizer.ros_callback)
        while not rospy.is_shutdown():
            if not visualizer.vis.poll_events():
                break
            else:
                visualizer.update_force(visualizer.wrench)
    else:
        # Static image mode – use existing runtime predictor utilities without ROS
        from shape_reconstruction.New_Camera import Camera
        from shape_reconstruction.New_Shape_Reconstruction import (
            initialise_depth_utils,
            raw_image_2_height_map,
        )
        from .generate_force_depth_dataset import simple_contact_mask, compute_depth_stats

        # Load cfg
        cfg_path = Path(args.cfg).resolve()
        import yaml
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg['sensor_id'] = args.sensor_id

        # Make calibration_root_dir absolute relative to YAML location if needed
        cal_root = Path(cfg['calibration_root_dir'])
        if not cal_root.is_absolute():
            cfg['calibration_root_dir'] = str((cfg_path.parent / cal_root).resolve())

        cam = Camera(cfg, calibrated=True, file_mode=True)
        initialise_depth_utils(cam, cfg)

        # Prepare LUT
        depth_dir = Path(cfg['calibration_root_dir']) / f"sensor_{args.sensor_id}" / cfg['depth_calibration']['depth_calibration_dir'].lstrip('/')
        p2d_iter = depth_dir / 'Pixel_to_Depth_iterative.npy'
        if p2d_iter.is_file():
            cam.Pixel_to_Depth = np.load(str(p2d_iter))
        else:
            from pathlib import Path as _P
            p2d_default = depth_dir / _P(cfg['depth_calibration']['Pixel_to_Depth_path']).name
            cam.Pixel_to_Depth = np.load(str(p2d_default))
        cam.max_index = cam.Pixel_to_Depth.shape[0] - 1
        cam.lighting_threshold = cfg.get('sensor_reconstruction', {}).get('lighting_threshold', 2)

        # Set reference frame
        cam.file_path = args.ref
        ref_bgr = cam.get_rectify_crop_image()
        cam.ref = ref_bgr
        cam.ref_GRAY = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

        # Target image
        cam.file_path = args.img
        bgr = cam.get_rectify_crop_image()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = simple_contact_mask(gray, threshold=5)

        # Depth stats and a simple scalar force estimate if coef is provided
        mean_d, _, _, _ = compute_depth_stats(cam, bgr, mask)
        if args.coef and Path(args.coef).is_file():
            with open(args.coef, 'r', encoding='utf-8') as f:
                params = json.load(f)
            intercept = float(params.get('intercept', 0.0))
            coef_mean = float(params.get('coef_mean_depth', 1.0))
            force_scalar = float(intercept + coef_mean * mean_d)
        else:
            force_scalar = float(mean_d)  # fallback: proportional to mean depth

        # Create a 3D visualization and save a screenshot
        visualizer = Visualizer()
        # Map scalar to XYZ components (use Z dominant)
        wrench = np.array([0.1 * force_scalar, 0.1 * force_scalar, force_scalar, 0, 0, 0])
        visualizer.update_force(wrench)
        visualizer.vis.update_renderer()
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        visualizer.vis.capture_screen_image(str(out_path))
        print(f"Saved static force visualization → {out_path}")

import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
from os import makedirs
import os
from os.path import exists, join
import shutil
import json
import open3d as o3d


class rs_reconstruct():
    
    def __init__(self, root_folder, record_imgs=True):
        self.root_folder = root_folder
        self.record_imgs = record_imgs
        self.color_raw_folder = os.path.join(self.root_folder, 'rgb_images')
        self.depth_raw_folder = os.path.join(self.root_folder, 'depth_images')
        self.pc_folder = os.path.join(self.root_folder, 'pc')

        # create the save folders 
        self.make_clean_folder(self.root_folder)
        self.make_clean_folder(self.color_raw_folder)
        self.make_clean_folder(self.depth_raw_folder)
        self.make_clean_folder(self.pc_folder)

        # create pipeline 
        self.pipeline = rs.pipeline()

        #  different resolutions of color and depth streams
        self.config = rs.config()

        # activate color and depth streams
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    def make_clean_folder(self, path_folder):
        if not exists(path_folder):
            makedirs(path_folder)

    def save_intrinsic_as_json(self, filename, frame):
        intrinsics = frame.profile.as_video_stream_profile().intrinsics
        with open(filename, 'w') as outfile:
            obj = json.dump(
                {
                    'width':
                        intrinsics.width,
                    'height':
                        intrinsics.height,
                    'intrinsic_matrix': [
                        intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                        intrinsics.ppy, 1
                    ]
                },
                outfile,
                indent=4)
        o3d_inter = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

        return o3d_inter

    def create_color_point_cloud(self, align_color_img, depth_img, 
                                depth_scale, clipping_distance_in_meters, intrinsic):
    
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(align_color_img.copy()), 
            o3d.geometry.Image(depth_img), 
            depth_scale=1.0/depth_scale,
            depth_trunc=clipping_distance_in_meters,
            convert_rgb_to_intensity = False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        flip_transform = [[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]]
        pcd.transform(flip_transform)

        return pcd.points, pcd.colors
                
    def start_streaming(self):

        # open3d visualizer 
        obj_pcd = o3d.geometry.PointCloud()
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480)

        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()

        # Using preset HighAccuracy for recording #3 for High Accuracy 
        depth_sensor.set_option(rs.option.visual_preset, 3)

        # depth sensor's depth scale
        depth_scale = depth_sensor.get_depth_scale()

        clipping_distance_in_meters = 8  # 8 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        align_to = rs.stream.color
        align = rs.align(align_to)

        # Streaming loop
        frame_count = 0
        try:
            while True:

                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                if self.record_imgs:
                    if frame_count == 0:
                        inter = self.save_intrinsic_as_json(
                            join(self.root_folder, "camera_intrinsic.json"), color_frame)  

                # get pointcloud 
                obj_pcd.points, obj_pcd.colors = self.create_color_point_cloud(color_image, depth_image, depth_scale, clipping_distance_in_meters, inter)
                         
                if self.record_imgs:
                    cv2.imwrite("%s/%06d.png" % \
                            (self.depth_raw_folder, frame_count), depth_image)
                    cv2.imwrite("%s/%06d.jpg" % \
                            (self.color_raw_folder, frame_count), color_image)
                    print("Saved color + depth image %06d" % frame_count)
                

                    o3d.io.write_point_cloud("%s/%06d.pcd" % \
                            (self.pc_folder, frame_count), obj_pcd)

                frame_count += 1

                # Render images
                depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
                images = np.hstack((color_image, depth_colormap))
                cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Recorder Realsense', images)

                if frame_count == 0:
                    vis.add_geometry(obj_pcd)
                vis.update_geometry(obj_pcd)
                vis.poll_events()
                vis.update_renderer()
                                  
                key = cv2.waitKey(1)

                # if 'esc' button pressed, escape loop and exit program
                if key == 27:
                    cv2.destroyAllWindows()
                    break

        finally:
            self.pipeline.stop()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Realsense Recorder. Please select one of the optional arguments")
    parser.add_argument("--output_folder",
                        default='../dataset/realsense/',
                        help="set output folder")

    args = parser.parse_args()

    save_path = args.output_folder

    rs_saver = rs_reconstruct(save_path, record_imgs=True)
    rs_saver.start_streaming()
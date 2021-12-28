'''
    https://dev.intelrealsense.com/docs/tensorflow-with-intel-realsense-cameras
'''

#
# import pyrealsense2 as rs # realsense
# import numpy as np
# import cv2
#
# # 깊이와 컬러 스트리밍 세팅
# pipeline = rs.pipeline()
# #print(rs_pipeline)
# config = rs.config()
# #print(config)
#
# # 스트리밍
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # depth
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # color
#
# aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
# point_cloud = rs.pointcloud()
#
# frames = pipeline.wait_for_frames()
# color_frame = frames.get_color_frame()
# color_image = np.asanyarray(color_frame.get_data())
#
#
# # obj_points = verts[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])].reshape(-1, 3)
# # zs = obj_points[:, 2]
# # ys = obj_points[:, 1]
'''
이미지 노이즈 제거 축소 팽창#1280 × 800
https://blog.daum.net/geoscience/1316
'''

import pyrealsense2 as rs # realsense
import numpy as np
import cv2

# 깊이와 컬러 스트리밍 세팅
rs_pipeline = rs.pipeline()
#print(rs_pipeline)
config = rs.config()
#print(config)

# 스트리밍
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # color
# 녹화 저장 ('파일형식')
#config.enalbe_record_to_file('object_detection.bag')
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('output.mp4', fourcc, 30, (640, 480))

# 스트리밍 시작
rs_pipeline.start(config)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

while True:
    # pipeline 으로 frame 받아오기
    frame = rs_pipeline.wait_for_frames()
    if not frame:
        break

    # pipeline frame에서 np배열로된 depth frame만 받아오기
    depth_frame = frame.get_depth_frame()
    # pipeline frame에서 np배열로된 color frame을 받아오기
    color_frame = frame.get_color_frame()
    # print("color frame : ",color_frame) # <pyrealsense2.frame BGR8 #frame> 출력

    # color frame을 np array 형태의 color image로 변환
    color_image = np.asanyarray(color_frame.get_data())
    # depth frame을 np array 형태의 depth image로 변환
    depth_image = np.asanyarray(depth_frame.get_data())

    cv2.imshow('color_images', color_image)




    # depth이미지를 depth color(거리별 컬러)로 변환
    # 각각의 값을 절대값화시키고 정수화, alpha 거리수치가 높은(가까운. 최대근접 1.0) 값중에 0.2까지만 남기기 (즉 더 가까운범위를 쪼개서거리측정가능)
    depth_alpha = cv2.convertScaleAbs(depth_image, alpha=0.2)
    #print(depth_alpha) # 3차원배열 색상값을 출력
    depth_colorImg = cv2.applyColorMap(depth_alpha, cv2.COLORMAP_JET) # COLORMAP_JET 컬러맵
    # depth이미지를 화면에 띄우기
    cv2.imshow('depth_images', depth_colorImg)

    depth_grayscale = cv2.cvtColor(depth_colorImg, cv2.COLOR_RGBA2GRAY)
    cv2.imshow('depth_grayscale_images', depth_grayscale)


    # 노이즈제거 부분. 
    white_mask = cv2.inRange(depth_grayscale, 100, 255)  # 범위내의 픽셀들은 흰색, 나머지 검은색
    whitefilter_images = cv2.bitwise_and(depth_grayscale, depth_grayscale, mask=white_mask)
    # 
    fgmask_images = cv2.morphologyEx(whitefilter_images, cv2.MORPH_OPEN, kernel)
    cv2.imshow('fgmask', fgmask_images)
    ## 작동확인\


    ret = cv2.waitKey(10)
    # ESC를 입력받을경우
    if ret == 27:
        break

# cv2 모든 창닫기
cv2.destroyAllWindows()
# 녹화 저장 끝내기
#out.release()
# 스트리밍 끝
rs_pipeline.stop()
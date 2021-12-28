'''
target_x와 target_y의 거리값을 print하는 코드

d435i 의 카메라에서 rgb이미지를 depth이미지를 각각 받아보는 코드
i부분 6축 센서 읽어들이는 부분이 아직 안들어간 형태의 코드

pip isntall pyrealsense2
pip install numpy #이미 설치되어있는지 확인
python-opencv #도 이미 설치되어있는지 확인
'''

import pyrealsense2 as rs # realsense
import numpy as np
import cv2


# 검정노이즈제거 white만 남길때
lower_color = 100  # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
upper_color = 255

# STEP 1. 카메라 열기
# 깊이와 컬러 스트리밍 세팅
rs_pipeline = rs.pipeline()
#print(rs_pipeline)
config = rs.config()
#print(config)

# STEP 2. 카메라 연결 
# 스트리밍
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # color
# 녹화 저장 ('파일형식')
#config.enalbe_record_to_file('object_detection.bag')
# 스트리밍 시작
rs_pipeline.start(config)

# STEP 3. 프레임 받아오기
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
    # print("color image : ", color_image) # 3차원 np배열형태의 이미지데이터가 출력

    # frame이미지를 화면에 띄우기
    #cv2.imshow('color_images', color_image)



    # depth이미지를 depth color(거리별 컬러)로 변환
    # 각각의 값을 절대값화시키고 정수화, alpha 거리수치가 높은(가까운. 최대근접 1.0) 값중에 0.2까지만 남기기 (즉 더 가까운범위를 쪼개서거리측정가능)
    depth_alpha = cv2.convertScaleAbs(depth_image, alpha=0.2)
    #print(depth_alpha) # 3차원배열 색상값을 출력
    depth_colorImg = cv2.applyColorMap(depth_alpha, cv2.COLORMAP_JET) # COLORMAP_JET 컬러맵

    # # depth이미지를 화면에 띄우기
    # cv2.imshow('depth_images', depth_colorImg)

    # STEP 4. 영상처리
    # depth이미지 흑백처리
    depth_grayscale = cv2.cvtColor(depth_colorImg,cv2.COLOR_RGBA2GRAY)
    cv2.imshow('depth_grayscale_images', depth_grayscale)

    # canny 처리 곡선첨예화
    #depth_canny = cv2.Canny(depth_grayscale, 100, 255)
    #cv2.imshow('depth_canny_images', depth_canny)

    key = cv2.waitKey(10)
    # 를 입력받을경우
    if key == 27:
        break

# STEP 5. 종료
# cv2 모든 창닫기
cv2.destroyAllWindows()
# 스트리밍 끝
rs_pipeline.stop()


''' 시간측정. 왜 필요함?
t1 = cv2.getTickCount();
#my_func(); // do something
t2 = cv2.getTickCount();
time = (t2 - t1) / cv2.getTickFrequency(); # 작동시간/초당 1틱으로 나누어, 매초마다 cv2가 초단위 수행시간측정
if time > 10: # 10초이상되면 종료
    break
'''
'''
    # 좌우로 붙여서 출력 np.vstack는 세로
    images = np.hstack((color_image, depth_colormap))
'''
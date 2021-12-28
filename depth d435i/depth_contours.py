import pyrealsense2 as rs # realsense
import numpy as np
import cv2

h, w = 480, 640
frame = 30
# depth 노이즈비율 최소값지정 최대값255 *white만 남길때
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
config.enable_stream(rs.stream.depth, w, h, rs.format.z16, frame) # depth
config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, frame) # color
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

    # pipeline에서 np배열로된 컬러 프레임만 받아오기
    color_frame = frame.get_color_frame()       # <pyrealsense2.frame BGR8 #frame> 출력
    # pipeline에서 np배열로된 뎁스 프레임만 받아오기
    depth_frame = frame.get_depth_frame()

    # color frame을 numpy array 형태의 컬러 이미지로 변환
    color_image = np.asanyarray(color_frame.get_data())
    # depth frame을 numpy array 형태의 뎁스 이미지로 변환
    depth_image = np.asanyarray(depth_frame.get_data())   # 400~4000 거리정보가 포함된 depth데이터 출력
    # print("color image : ", color_image) # 3차원 np배열형태의 이미지데이터가 출력

    # 컬러 이미지를 화면에 띄우기
    #cv2.imshow('color_images', color_image)

    # depth이미지를 depth color(거리별 컬러)로 변환
    # 각각의 값을 절대값화시키고 정수화, alpha 거리수치가 높은(가깝게보려면 최대근접 1.0) 값중에 0.2까지만 남기기 (즉 더 가까운범위를 쪼개서거리측정가능)
    depth_alpha = cv2.convertScaleAbs(depth_image, alpha=0.15)
    #print(depth_alpha) # 3차원배열 색상값을 출력
    depth_colorImg = cv2.applyColorMap(depth_alpha, cv2.COLORMAP_JET) # COLORMAP_JET 컬러맵

    # # depth이미지를 화면에 띄우기
    # cv2.imshow('depth_images', depth_colorImg)


    # STEP 4. 전후처리
    # depth이미지 흑백처리
    depth_grayscale = cv2.cvtColor(depth_colorImg,cv2.COLOR_RGBA2GRAY)
    #cv2.imshow('depth_grayscale_images', depth_grayscale)

    # 특정 white만 남기기 최소lower, 최대upper
    #white_mask = cv2.inRange(depth_grayscale, lower_color, upper_color)  # 범위내의 픽셀들은 흰색, 나머지 검은색
    #whitefilter_images = cv2.bitwise_and(depth_grayscale, depth_grayscale, mask=white_mask)
    #cv2.imshow('white_result', whitefilter_images)

    # 영상 이진화 OTSU
    _, OTSU_binary = cv2.threshold(depth_grayscale, 0, 255, cv2.THRESH_OTSU)

    # 잡음제거
    # 가우시안 블러와 OTSU로 노이즈제거
    OTSU_blur = cv2.GaussianBlur(OTSU_binary, (5, 5), 0)
    cv2.imshow("OTSU_blur", OTSU_blur)
    ret, OTSU_gaubin = cv2.threshold(OTSU_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('src_gaubin', OTSU_gaubin)

    # STEP 5. 외곽선 검출
    # cv2 contours 외곽선 추출함수
    contours, _ = cv2.findContours(OTSU_gaubin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 검은판 하나 생성
    contours_images = np.zeros((h, w, 3), np.uint8)

    # 모든 객체 외곽선
    for Line_idx in range(len(contours)):
        color = (0,0,255)  # 빨강
        cv2.drawContours(contours_images, contours, Line_idx, color, 1, cv2.LINE_AA)

    flag=0
    # 외곽선 연결선
    for contour in contours:
        # convexHull 나머지 모든점을 포함하는 다각형을 만들어내는 알고리즘 = 장애물 외곽선
        conhull = cv2.convexHull(contour)
        # 다각형의 좌표값이 들어간 conhull axis(행으로 계산)해 x축끼리 y축끼리의 max min을 구하고 //결론np.max는 모서리만 출력한다.
        hull = np.max(conhull, axis=1) # 모서리좌표
        maxbox = np.max(hull, axis=1) # 한 노란박스 내에서 모서리 중 제일큰값들
        # 최대값x와 최소값x을 뺀 절대값이 (노란박스크기가)작은건 안그려지게한다 
        if abs(max(maxbox) - min(maxbox)) > 60: # 장애물이 60이상 큰것만 잡히도록
            cv2.drawContours(contours_images, [conhull], 0, (0, 255, 255), 3) # 노랑

            # 한 박스(hull) 마다  맨 좌측값, 맨우측값 가져오기
            max_y = np.array([24, 0]) # y축 최저값
            min_y = np.array([639, 0]) # y축 최대값
            for count in hull:
                if max_y[0] < count[0]:
                    max_y = count
                if min_y[0] > count[0]:
                    min_y = count
            #print("max",max_y)
            #print("min",min_y)
            cv2.circle(contours_images, max_y, 3, (255, 0, 0), 2, cv2.LINE_AA)  # 최우측 좌표에 파란
            cv2.circle(contours_images, min_y, 3, (255, 255, 0), 2, cv2.LINE_AA)  # 최좌측 좌표에 청록

            # 오른족에서 시작된 장애물
            if max_y[0] >= 639:
                print("rightObj_max", max_y)
                print("rightObj_min", min_y)
            # 왼쪽에서 시작된 장애물
            elif min_y[0] <= 28:
                print("leftObj_max", max_y) #최좌측값
                print("leftObj_min", min_y)
            # 좌우 연결구조가 없는 증앙에 있는 장애물
            #else:

            # 가져올좌표 점그려보기
            cv2.circle(contours_images, (max_y[1],max_y[0]-10), 5, (255, 0, 255), 2, cv2.LINE_AA) # 분홍
            #target_depth = depth_frame.get_distance( int(max_y[1]), int(max_y[0]-10) )
            #print("{:.2f} cm".format(100 * target_depth))


    cv2.imshow('contours_images', contours_images)
    # cv2.imshow('depth_grayscale_images', depth_grayscale)

    key = cv2.waitKey(10)
    # 를 입력받을경우
    if key == 27:
        break

# STEP 6. 종료
# cv2 모든 창닫기
cv2.destroyAllWindows()
# 스트리밍 끝
rs_pipeline.stop()


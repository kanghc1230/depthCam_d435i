import pyrealsense2 as rs # realsense
import numpy as np
import cv2

# parameter
h, w = 480, 640
frame = 30

class depth_stream:
    def __init__(self):

        # STEP 1. 카메라 열기
        # 깊이와 컬러 스트리밍 세팅 파이프라인 연결
        self.rs_pipeline = rs.pipeline()
        # 기본설정 가져오기
        self.config = rs.config()

        # STEP 2. 카메라 연결
        # config 설정을 수정. w,h, 특정형식으로 스트리밍
        self.config.enable_stream(rs.stream.depth, w, h, rs.format.z16, frame)  # depth
        self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, frame)  # color
        # 녹화 저장 코드 ('파일이름,형식')
        # config.enalbe_record_to_file('object_detection.bag')
        # 파이프라인 시작
        self.rs_pipeline.start(self.config)

        # pipeline 으로 frame 받아오기
        self.__main__()

    # 장애물의 왼쪽끝, 오른쪽 좌표값을 받고, 그 사이중 가장가까운(색상이높은) 거리값을 가져오는 함수
    def location_to_depth(self, loc1,loc2):
        # (오른쪽에서 시작한 장애물) 장애물 왼쪽끝좌표의 오른쪽으로 거리들을 읽어온다.
        if loc1 [0] < loc2[0]:
            arr = []
            start = loc1[0]
            end = loc2[0]
            # 장애물의 왼쪽끝부터 오른쪽끝까지 사이값을 numpy배열로
            for i in range(start, end+1):
                arr.append(self.depth_grayscale[loc1[1], i])
            np_arr = np.array(arr)
            index_arr = np.argmax(np_arr)
            # 가장높은값 표시
            #cv2.circle(contours_images, (loc1[0] + index_arr, loc1[1]), 3, (0, 0, 255), 5, cv2.LINE_AA)  # 빨강
            # 거리추출
            target_depth = self.depth_frame.get_distance( loc1[0] + index_arr, loc1[1] )

        return "{:.2f} cm".format(100 * target_depth)

    # pipeline으로 동영상 frame 받아오기
    def __main__(self):
        # STEP 3. 프레임 받아오기
        while True:
            frame = self.rs_pipeline.wait_for_frames()
            if not frame:
                break

            # pipeline에서 np배열로된 컬러 프레임만 받아오기
            self.color_frame = frame.get_color_frame()  # <pyrealsense2.frame BGR8 #frame> 출력
            # pipeline에서 np배열로된 뎁스 프레임만 받아오기
            self.depth_frame = frame.get_depth_frame()

            # color frame을 numpy array 형태의 컬러 이미지로 변환
            color_image = np.asanyarray(self.color_frame.get_data())
            # depth frame을 numpy array 형태의 뎁스 이미지로 변환
            depth_image = np.asanyarray(self.depth_frame.get_data())   # 400~4000 거리정보가 포함된 depth데이터 출력
            # print("color image : ", color_image) # 3차원 np배열형태의 이미지데이터가 출력

            # 컬러 이미지를 화면에 띄우기
            #cv2.imshow('color_images', color_image)

            # depth이미지를 depth color(거리별 컬러)로 변환
            # 각각의 값을 절대값화시키고 정수화, alpha 거리수치가 높은(가깝게보려면 최대근접 1.0) 값중에 0.15까지만 남기기 (즉 더 가까운범위를 쪼개서거리측정가능)
            depth_alpha = cv2.convertScaleAbs(depth_image, alpha=0.2)
            #print(depth_alpha) # 3차원배열 색상값을 출력
            depth_colorImg = cv2.applyColorMap(depth_alpha, cv2.COLORMAP_JET) # COLORMAP_JET 컬러맵

            # # depth이미지를 화면에 띄우기
            # cv2.imshow('depth_images', depth_colorImg)


            # STEP 4. 전후처리
            # depth이미지 흑백처리
            self.depth_grayscale = cv2.cvtColor(depth_colorImg,cv2.COLOR_RGBA2GRAY)
            #cv2.imshow('depth_grayscale_images', depth_grayscale)

            # 영상 이진화 OTSU
            _, OTSU_binary = cv2.threshold(self.depth_grayscale, 0, 255, cv2.THRESH_OTSU)

            # 잡음제거
            # 가우시안 블러와 OTSU로 노이즈제거
            OTSU_blur = cv2.GaussianBlur(OTSU_binary, (5, 5), 0)
            cv2.imshow("OTSU_blur", OTSU_blur)
            ret, OTSU_gaubin = cv2.threshold(OTSU_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imshow('src_gaubin', OTSU_gaubin)

            # STEP 5. 외곽선 검출
            # cv2 contours 외곽선 추출함수
            contours, _ = cv2.find\
                (OTSU_gaubin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            # 검은판 하나 생성
            contours_images = np.zeros((h, w, 3), np.uint8)

            # 모든 객체 외곽선
            for Line_idx in range(len(contours)):
                color = (0,0,255)  # 빨간선
                cv2.drawContours(contours_images, contours, Line_idx, color, 1, cv2.LINE_AA)

            # 외곽선 연결선
            box_cnt = 0
            for contour in contours:
                # convexHull 나머지 모든점을 포함하는 다각형을 만들어내는 알고리즘 = 장애물 외곽선
                conhull = cv2.convexHull(contour)
                # 다각형의 좌표값이 들어간 conhull axis(행으로 계산)해 x축끼리 y축끼리의 max min을 구하고 //결론 np.max는 모서리만 출력한다.
                hull = np.max(conhull, axis=1)  # 모서리좌표
                maxbox = np.max(hull, axis=1)  # 한 노란박스 내에서 모서리 중 제일큰값들
                # 최대값x와 최소값x을 뺀 절대값이 (노란박스크기가)작은건 안그려지게한다
                if abs(max(maxbox) - min(maxbox)) > 90:  # 장애물이 60이상 큰것만 잡히도록
                    cv2.drawContours(contours_images, [conhull], 0, (0, 255, 255), 3)  # 노랑

                    # 한 박스(hull) 마다  맨 좌측값, 맨우측값 가져오기
                    max_y = np.array([24, 0])  # y축 맨오른쪽좌표가 들어갈 변수
                    min_y = np.array([639, 0])  # y축 맨왼쪽좌표가 들어갈 변수
                    # 박스hull의 좌표중 벽(화면끝)과 닿아있는 구조물이라면
                    for count in hull:
                        if max_y[0] < count[0]:
                            max_y = count
                        if min_y[0] > count[0]:
                            min_y = count
                    cv2.circle(contours_images, max_y, 3, (255, 0, 0), 2, cv2.LINE_AA)  # 최우측 좌표에 파란
                    cv2.circle(contours_images, min_y, 3, (255, 255, 0), 2, cv2.LINE_AA)  # 최좌측 좌표에 청록

                    # 장애물의 거리추출함수 왼쪽좌표값, 0번(오른쪽시작)
                    depth = self.location_to_depth(min_y, max_y)
                    #print(box_cnt,"번째 장애물의 거리값 = " ,depth, " cm")

                    label = "obs." + str(box_cnt) + " depth=" + str(depth)
                    cv2.rectangle(contours_images, min_y-(3,10), min_y+(120,5), (100, 255, 100), cv2.FILLED)
                    cv2.putText(contours_images, label, min_y, cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 0), 1)

                    box_cnt = box_cnt + 1

            cv2.imshow('contours_images', contours_images)
            # cv2.imshow('depth_grayscale_images', depth_grayscale)

            cv2.imshow('color_images', color_image)

            key = cv2.waitKey(10)
            # 를 입력받을경우
            if key == 27:
                break

        # 스트리밍 끝
        self.rs_pipeline.stop()

if __name__ == '__main__':
    cal1 = depth_stream()
    # STEP 6. 종료
    # cv2 모든 창닫기
    depth_stream
    cv2.destroyAllWindows()



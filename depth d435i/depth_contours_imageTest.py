import numpy as np
import cv2
import sys


# 장애물의 왼쪽끝, 오른쪽 좌표값을 받으면 가장 가까운 좌표가 나오는함수
def location_to_depth(loc1,loc2):
    # (오른쪽에서 시작한 장애물) 장애물 왼쪽끝좌표의 오른쪽으로 거리들을 읽어온다.
    if loc1[0] < loc2[0]:
        #cv2.circle(contours_images, (loc[0] + 10, loc[1]), 3, (255, 0, 255), 1, cv2.LINE_AA)  # 분홍
        arr = []
        start = loc1[0]
        end = loc2[0]
        for i in range(start, end+1):
            arr.append(depth_grayscale[loc1[1], i])
        np_arr = np.array(arr)
        index_arr = np.argmax(np_arr)
        cv2.circle(contours_images, (loc1[0] + index_arr, loc1[1]), 3, (0, 0, 255), 5, cv2.LINE_AA)  # 분홍
    return loc1[0] + index_arr, loc1[1]

h, w = 480, 640

src = cv2.imread("contours_output2.png", cv2.IMREAD_COLOR)

depth_grayscale = cv2.cvtColor(src,cv2.COLOR_RGBA2GRAY)
if src is None:
    print("can't read file")
    sys.exit

cv2.imshow("src", src)

_, OTSU_binary = cv2.threshold(depth_grayscale, 0, 255, cv2.THRESH_OTSU)

# 잡음제거
# 가우시안 블러와 OTSU로 노이즈제거
OTSU_blur = cv2.GaussianBlur(OTSU_binary, (5, 5), 0)
cv2.imshow("OTSU_blur", OTSU_blur)
ret, OTSU_gaubin = cv2.threshold(OTSU_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('src_gaubin', OTSU_gaubin)



a = (np.where(OTSU_gaubin > 200 ))
print(a)

# STEP 5. 외곽선 검출
# cv2 contours 외곽선 추출함수
contours, _ = cv2.findContours(OTSU_gaubin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# 검은판 하나 생성
contours_images = np.zeros((h, w, 3), np.uint8)

# 모든 객체 외곽선
for Line_idx in range(len(contours)):
    color = (0,0,255)  # 빨강
    cv2.drawContours(contours_images, contours, Line_idx, color, 1, cv2.LINE_AA)


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
        max_y = np.array([24, 0]) # y축 맨오른쪽좌표가 들어갈 변수
        min_y = np.array([639, 0]) # y축 맨왼쪽좌표가 들어갈 변수
        for count in hull:
            if max_y[0] < count[0]:
                max_y = count
            if min_y[0] > count[0]:
                min_y = count
        cv2.circle(contours_images, max_y, 3, (255, 0, 0), 2, cv2.LINE_AA)  # 최우측 좌표에 파란
        cv2.circle(contours_images, min_y, 3, (255, 255, 0), 2, cv2.LINE_AA)  # 최좌측 좌표에 청록

        # 오른쪽에서 시작된 장애물
        if max_y[0] >= 639:
            print("rightObj_max_y", max_y)
            print("rightObj_min_y", min_y)

        # 왼쪽에서 시작된 장애물
        elif min_y[0] <= 28:
            print("leftObj_max_y", max_y)  # 최좌측값
            print("leftObj_min_y", min_y)

        # 좌우 연결구조가 없는 증앙에 있는 장애물
        else:
            print("centerObj_max_y", max_y)
            print("centerObj_min_y", min_y)

        # 장애물의 거리추출함수 왼쪽좌표값, 0번(오른쪽시작)
        location_to_depth(min_y, max_y)

cv2.imshow('contours_images', contours_images)

cv2.waitKey(0)

cv2.destroyAllWindows()
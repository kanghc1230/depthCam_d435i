'''
 화면의 색상기준으로 가장자리 검출하는 예제

'''

import sys
import cv2
import numpy as np
import random

src = cv2.imread("contours_output2.png",cv2.IMREAD_GRAYSCALE)
if src is None:
    print("image load failed")
    sys.exit(1)

# 영상 이진화 OTSU
_, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)

# 가우시안 블러와 OTSU로 노이즈제거
src_blur = cv2.GaussianBlur(src_bin, (5,5), 0)
cv2.imshow("src_blur", src_blur)
ret, src_gaubin = cv2.threshold(src_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('src_gaubin', src_gaubin)

contours, _ = cv2.findContours(src_gaubin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# for contours_list in contours:
#     print(contours)


h, w = src.shape[:2]
dst = np.zeros((h,w,3), np.uint8)

for Line_idx in range(len(contours)):
    # 특정색상결정
    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    # 선 그리기
    cv2.drawContours(dst, contours, Line_idx , color, 1, cv2.LINE_AA)

    contours_list = np.concatenate(contours).tolist()
    contours_list2 = np.concatenate(contours_list).tolist()
    maxCont = 0
    for contour in contours_list2:
        if maxCont < contour[1]:
            maxCont = contour[1]
            print()
    contour
    #print(maxCont)

cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()
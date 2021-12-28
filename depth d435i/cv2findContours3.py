import cv2
import numpy as np


img_color = cv2.imread('contours_output2.png')
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
ret, img_binary = cv2.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


for cnt in contours:
    cv2.drawContours(img_color, [cnt], 0, (255, 0, 0), 3)  # blue
cv2.imshow("result", img_color)
cv2.waitKey(0)


for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(img_color, [hull], 0, (255, 0, 255), 5)
cv2.imshow("result2", img_color)
cv2.waitKey(0)


for cnt in contours:
    hull = cv2.convexHull(cnt, returnPoints=False)
    print(hull, cnt)
    defects = cv2.convexityDefects(cnt, hull)
    print(defects)

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        cv2.line(img_color, start, end, [0, 0, 255], 2)

        cv2.imshow("result3", img_color)
        cv2.waitKey(0)
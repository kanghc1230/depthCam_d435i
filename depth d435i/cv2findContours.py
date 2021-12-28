'''
 화면의 색상기준으로 가장자리 검출하는 예제

'''

import cv2

src = cv2.imread("contours.png", cv2.IMREAD_COLOR)
cv2.imshow("src1", src)

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray", gray)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

#contours, hierachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#cv2.drawContours(src, [contours[0]], 0, (0, 0, 255), 2)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print("contours = ", contours)
for cnt in contours:
    cv2.drawContours(src, [cnt], 0, (0, 0, 255), 3)  # red

cv2.imshow("src2", src)
cv2.waitKey(0)

cv2.destroyAllWindows()
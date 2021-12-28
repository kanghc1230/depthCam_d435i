import cv2
import sys

# 비디오 파일 열기
#cap = cv2.VideoCapture('camshift.avi')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 초기 사각형 영역: (x, y, w, h)
# ROI로 선택해도 되지만 강제로 입력함
x, y, w, h = 135, 220, 100, 100
rc = (x, y, w, h)

# 영상의 정보 받아오기
ret, frame = cap.read()

if not ret:
    print('frame read failed!')
    sys.exit

roi = frame[y:y + h, x:x + w]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# HS 히스토그램 계산
channels = [0, 1]  # H와 S만 이용. V는 안씀
ranges = [0, 180, 0, 256]
hist = cv2.calcHit([roi_hsv], channels, None, [90, 128], ranges)

# Mean Shift 알고리즘 종료 기준
term_crit = (cv2.TERM_CRITERIA_EPA | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # HS 히스토그램에 대한 역투영
    # frame을 HSV로 변환
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 히스토그램 역투영 확률 데이터 얻기
    backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)

    # Mean Shift
    # 역투영 확률값을 Mean shift 인자에 입력
    _, rc = cv2.meanShift(backproj, rc, term_crit)

    # 추적 결과 화면 출력
    cv2.rectangle(frame, rc, (0, 0, 255), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(60) == 27:
        break

cap.release()
cv2.destroyAllWindows()
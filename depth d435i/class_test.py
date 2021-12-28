import cv2

class depth_stream:
    def __init__(self):
        print("init")
        self.depth = 1
        self.__main__()

    def location_to_depth(self, loc1, loc2):
        if loc1[0] < loc2[0]:
            print("ltd")

    def __main__(self):
        print("main")
        frame = cv2.imread("contours_output2.png", cv2.IMREAD_COLOR)
        cv2.imshow("contours", frame)
        cv2.waitKey(0)
        print(self.depth)
        # 를 입력받을경우

if __name__ == '__main__':
    test = depth_stream()
    # STEP 6. 종료
    # cv2 모든 창닫기
    cv2.destroyAllWindows()
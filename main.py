from FaceExpressDec import faceExpressDec
import cv2


if __name__ == '__main__':
    # 开启摄像头
    cap = cv2.VideoCapture(0)
    # 摄像头大小
    cap.set(3, 256)
    cap.set(4, 256)

    # 使用表情识别类
    fed=faceExpressDec(cap)

    #测试使用
    flag, frame = cap.read()
    print("摄像头状态:", flag)
    while(flag):
        flag, img = cap.read()
        cv2.imshow("face detection landmark", img)
        out=fed(img)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(out)
    cap.release()
    cv2.destroyAllWindows()


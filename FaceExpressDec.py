# ================================================================
#
#   Editor      : Pycharm
#   File name   : FaceExpressDec
#   Author      : LLLLLuuusa(HuangDaxu)
#   Created date: 2022-04-27 20:57
#   Email       : 1095663821@qq.com
#   QQ          : 1095663821
#   Description :
#
#     (/≧▽≦)/ long mine the sun shine!!!
# ================================================================
import cv2
import dlib
import asyncio
import numpy as np
from PIL import Image,ImageFont,ImageDraw

class faceExpressDec():
    def __init__(self,isShow=True,isDebug=False,modelPath="model/shape_predictor_68_face_landmarks.dat",fontpath = r"C:\Windows\Fonts\simkai.ttf"):

        #是否本地计算机显示摄像头内容(在屏幕上开窗口，查看摄像头内容）
        self.isShow = isShow

        #是否开启调试模式
        self.isDebug = isDebug


        #加载对应的模型
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(modelPath)

        #配置字体
        self.font = ImageFont.truetype(fontpath, 36)


    def dec(self,img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray, 1)

        out = None
        # 绘制每一个人脸
        for face in faces:
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 3)
            face_width = face.right() - face.left()
            face_higth = face.top() - face.bottom()
            shape = self.predictor(gray, face)
            img , out = self.express(shape.parts(), face_width, face_higth, img)
            # #绘制每一个关键点
            # for pt in shape.parts():
            #     pt_position = (pt.x, pt.y)
            #     cv2.circle(img, pt_position, 3, (0, 0, 255), -1)

        # 窗口页面
        if self.isShow:
            cv2.imshow("face detection landmark", img)
        return out

    def express(self,pts, face_width, face_higth, frame):
        isDubug = self.isDebug
        out = None

        mouth_width = (pts[54].x - pts[48].x) / face_width  # 嘴巴咧开程度
        if isDubug:
            print("嘴角张开程度", mouth_width)  # 正常<0.36   嘴角轻轻上扬0.36-0.39    嘴角张开>0.39
        mouth_higth = (pts[66].y - pts[62].y) / -face_higth  # 嘴巴张开程度
        if isDubug:
            print("嘴巴张开程度", mouth_higth)  # 嘴巴张开 >0.1
        eyes_higth = ((pts[37].y + pts[38].y + pts[43].y + pts[44].y) - (
                    pts[41].y + pts[40].y + pts[47].y + pts[46].y)) / face_higth  # 眼睛睁开程度
        if isDubug:
            print("眼睛睁开程度", eyes_higth)  # 双眼睁开 >0.1

        # 眉毛直线拟合数据缓冲
        line_brow_x = []
        line_brow_y = []
        # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
        brow_hight = 0  # 高度之和
        brow_width = 0  # 两边眉毛距离之和
        for j in range(17, 21):
            brow_hight += (pts[j].y + pts[j + 5].y) / -face_higth
            brow_width += pts[j + 5].x - pts[j].x
            line_brow_x.append(pts[j].x)
            line_brow_y.append(pts[j].y)

        tempx = np.array(line_brow_x)
        tempy = np.array(line_brow_y)
        z1 = np.polyfit(tempx, tempy, 1)  # 拟合成一次直线
        brow_k = -round(z1[0], 3)  # 拟合出曲线的斜率和实际眉毛的倾斜方向是相反的
        if isDubug:
            print("眉毛弯曲程度", brow_k)  # 皱眉<0 舒眉>0

        img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        # 笑 眉毛>0.1,嘴巴>0.1,嘴角>0.39
        if (brow_k >= 0.1 and mouth_higth >= 0.1 and mouth_width >= 0.39):
            if isDubug:
                print("笑")
            draw.text((40, 40), "笑", font=self.font, fill=(255, 0, 0))
            out="笑"

        # 微笑 眉毛>0,嘴巴<0.1,嘴角<0.36-0.39,眼睛>0.1
        elif (
                brow_k >= 0.1 and mouth_higth <= 0.1 and mouth_width >= 0.36 and mouth_width <= 0.39 and eyes_higth > 0.1):
            if isDubug:
                print("微笑")
            draw.text((40, 40), "微笑", font=self.font, fill=(255, 0, 0))
            out="微笑"

        # 思考 眉毛<0,嘴巴<0.1,嘴角<0.36,眼睛>0.1
        elif (brow_k < 0.1 and mouth_higth < 0.1 and mouth_width < 0.36 and eyes_higth > 0.1):
            if isDubug:
                print("思考")
            draw.text((40, 40), "思考", font=self.font, fill=(255, 0, 0))
            out="思考"

        # 思考 眉毛>0,嘴巴<0.1,嘴角<0.36,眼睛>0.1
        elif (brow_k > 0.1 and mouth_higth < 0.1 and mouth_width < 0.36 and eyes_higth > 0.1):
            if isDubug:
                print("正常")
            draw.text((40, 40), "正常", font=self.font, fill=(255, 0, 0))
            out="正常"

        # 愤怒 眉毛<0,嘴巴>0.1,嘴角<0.36,眼睛>0.1
        elif (brow_k < 0.1 and mouth_higth > 0.1 and mouth_width < 0.39 and eyes_higth > 0.1):
            if isDubug:
                print("愤怒")
            draw.text((40, 40), "愤怒", font=self.font, fill=(255, 0, 0))
            out="愤怒"
        # 转换回OpenCV格式
        img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

        return img,out

    def __call__(self,img):
        #默认开启
        out = self.dec(img)

        return out


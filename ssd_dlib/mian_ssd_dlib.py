# 单进程，使用dnn调用ssd模型识别目标，使用dlib进行目标追踪
# from utils import FPS
import numpy as np
import dlib
import cv2
import datetime

# 计算帧率
class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


# SSD标签
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# 模型分数最低标准
g_confidence = 0.1

if __name__ == '__main__':
    # 读取网络模型
    print("加载模型")
    net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                                    "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

    # 初始化
    print("开始加载视频")
    vs = cv2.VideoCapture("race.mp4")
    writer = None

    trackers = [] # 需要追踪的目标
    labels = [] # 目标标签

    # 计算FPS帧率
    fps = FPS()
    fps.start()

    while True:
        # 读取一帧
        (graded, frame) = vs.read()

        # 最后一帧退出
        if frame is None:
            break

        # 预处理
        (h,w) = frame.shape[:2]
        width = 600
        r = width / float(w)
        height = int(h * r)
        dim = (width,height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 由 BGR 转为 RGB，算法模型需要

        # 先检测由哪些物体，再追踪
        if len(trackers) == 0: # 需追踪物体为零，即未进行ssd识别目标
            # 获取blob数据
            (h,w) = frame.shape[:2]

            # 均一化
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (w,h), 127.5)

            # 得到检测结果
            net.setInput(blob)
            detections = net.forward()

            # 遍历得到的检测结果
            for i in np.arange(0, detections.shape[2]):
                # 能检测到多个结果，只保留概率高的
                confidence = detections[0, 0, i, 2] # ssd模型识别的相似度分数

                # 过滤
                if confidence > g_confidence:
                    # 获取识别物体的标签
                    idx = int(detections[0, 0, i, 1])
                    label = CLASSES[idx]

                    # 只保留标签为人的
                    if CLASSES[idx] != "person":
                        continue

                    # 获取识别到物体的框
                    # 返回的是 h 和 w 的比例位置
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # 使用dlib来进行目标追踪
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(startX),int(startY),int(endX),int(endY))
                    tracker.start_track(rgb, rect)

                    # 保存结果
                    labels.append(label)
                    trackers.append(tracker)

                    # 绘图
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0),2)
                    cv2.putText(frame, label, (startX, startY-15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0,255,0),2)

        else: # 以进行ssd目标识别，直接开始追踪
            # 每一个追踪器都要进行更新
            for (t,l) in zip(trackers, labels):
                tracker.update(rgb)
                pos = tracker.get_position() # 直接从dlib返回的是图像坐标，无需转换

                # 得到位置
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # 绘图
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0),2)
                cv2.putText(frame, label, (startX, startY-15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0,255,0),2)


        # 显示
        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        # 计算FPS
        fps.update() # 帧数加一

    fps.stop()
    print("elepsed time: {:.2f}".format(fps.elapsed()))
    print("approx. FPS: {:2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.release()
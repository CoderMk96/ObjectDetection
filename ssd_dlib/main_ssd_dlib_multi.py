# 多进程，使用dnn调用ssd模型识别目标，使用dlib进行目标追踪
import multiprocessing
import numpy as np
import dlib
import cv2
import datetime
# from utils import FPS

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


# 每个进程的函数，对目标进行跟踪
def start_tracker(box, label, rgb, inputQueque, outputQueque):
    tracker = dlib.correlation_track()
    rect = dlib.rectangele(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    tracker.start_track(rgb, rect)

    while True:
        # 获取下一帧
        rgb = inputQueque.get()

        # 非空就开始处理
        if rgb is not None:
            # 更新追踪器
            tracker.update(rgb)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # 把结果放到输出q
            outputQueque.put((label, (startX, startY, endX, endY)))


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


if __name__ == '__main__':

    inputQueues = []
    outputQueues = []

    print("[INFO] 读取算法模型")
    net = cv2.dnn.readNetFromCaffe("D:\code_study\opencv_study\ObjectDetection\ssd_dlib\mobilenet_ssd\MobileNetSSD_deploy.prototxt",
                                   "D:\code_study\opencv_study\ObjectDetection\ssd_dlib\mobilenet_ssd\MobileNetSSD_deploy.caffemodel")

    print("[INFO] 开始播放流媒体")
    vs = cv2.VideoCapture("race.mp4")
    writer = None

    fps = FPS().start()

    while True:
        (grabbed, frame) = vs.read()

        if frame is None:
            break

        (h, w) = frame.shape[:2]
        width = 600
        r = width / float(w)
        dim = (width, int(h * r))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 先检测位置
        if len(inputQueues) == 0:
            (h, w) = frame.shape[:2]
            # 均一化：缩放因子是 0.007843
            blob = cv2.dnn.blobFromImage(frame, 0.7843, (w,h), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                print("[INFO] confidence: {:.2f}".format(confidence))
                # 匹配分数需要大于0.2
                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])
                    label = CLASSES[idx]

                    print("[INFO] label: {}".label)
                    if CLASSES[idx] != "person":
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    pos = (startX, startY, endX, endY)

                    # 创建了两个多进程队列 iq 和 oq
                    iq = multiprocessing.Queue()
                    oq = multiprocessing.Queue()
                    inputQueues.append(iq)
                    outputQueues.append(oq)

                    # 创建了一个多进程 Process 对象
                    process = multiprocessing.Process(
                        target=start_tracker,
                        args=(pos, label, rgb, iq, oq)
                    )
                    process.deamon = True # 将该进程设置为守护进程，意味着当主进程结束时，该进程也会随之结束
                    process.start()

                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        else:
            # 多个追踪器处理的都是相同输入
            for iq in inputQueues:
                iq.put(rgb)

            for oq in outputQueues:
                # 得到更新结果
                (label, (startX, startY, endX, endY)) = oq.get()

                # 绘图
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.release()









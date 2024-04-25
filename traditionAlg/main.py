# 单进程，需要用户手动框选待追踪物体，通过kcf追踪算法实现多目标追踪
import time
import cv2
import numpy as np

trackerName = "kcf" # opencv实现的8种追踪算法，其中之一

# opencv已经实现了的追踪算法
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}


if __name__ == '__main__':
    # 实例化OpenCV的多目标追踪对象
    trackers = cv2.MultiTracker_create()
    vs = cv2.VideoCapture("videos/soccer_01.mp4")

    while True:
        frame = vs.read() # 包含[true, data]
        frame = frame[1]

        # 结束(最后一帧)
        if frame is None:
            break

        # resize每一帧
        (h,w) = frame.shape[:2]
        width = 600
        r = width/ float(w)
        dim = (width, int(h*r))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # 开始追踪，并获取结果
        (success, boxes) = trackers.update(frame)

        # 绘制区域
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,0),2)

        # 显示
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(100) & 0xFF

        if key == ord("s"):
            # 选择一个区域
            box = cv2.selectROI("Frame",frame,fromCenter=False,showCrosshair=True)

            # 创建一个新的追踪器
            tracker = OPENCV_OBJECT_TRACKERS[trackerName]( )
            trackers.add(tracker,frame,box)
        elif key == 27:
            break

vs.release()
cv2.destroyAllWindows()



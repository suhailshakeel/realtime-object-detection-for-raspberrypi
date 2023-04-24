import numpy as np
#import imutils
import time
import cv2
import threading
from collections import deque
data = []
key = 1
net = cv2.dnn.readNet('C:/Users/salro/Desktop/Final Project/yolov4-tiny.weights', 'C:/Users/salro/Desktop/Final Project/yolov4-tiny.cfg')

classes = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant"
,"stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack"
,"umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove"
,"skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich"
,"orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet"
,"tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book"
,"clock","vase","scissors","teddy bear","hair drier","toothbrush"]

def dandr():

    # Set up webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = cap.read()

        # Create blob from input frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
        # print(blob)

        # Set input to the YOLOv4-Tiny network
        net.setInput(blob)

        # Forward pass through network
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        # print(output_layers)
        outs = net.forward(output_layers)

        # Process detection results
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                #area of item
                area = w * h

    
                #queue size is 50
                if len(data) > 50:
                    data.pop(0)
                data.append([label, area])
        global key
        if cv2.waitKey(1) == ord('q'):
            break


        time.sleep(0.1)#10 fps

t = threading.Thread(target=dandr,)

def start():
    global key
    key = 1
    t.start()


def stop():
    global key
    key = 0

def fs_unique(data, items):
    #Finding Unique Items
    lst = []
    high = -1
    for i in items:
        for j in data:
            if (i == j[0]) and (high < j[1]):
                high = j[1]
        
        for j in data:
            if j[1] == high:
                lst.append(j)
                break
        high = 0
    
    #Sorting Unique Items
    for i in range(len(lst)):
        for j in range(0, len(lst) - i - 1):
            if lst[j][1] < lst[j + 1][1]:
                temp = lst[j]
                lst[j] = lst[j+1]
                lst[j+1] = temp
        
    return lst

if __name__ == "__main__":
    start()
    time.sleep(10)
    print(fs_unique(data, classes))
    stop()
    print(1)
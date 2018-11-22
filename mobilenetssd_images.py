import xml.etree.cElementTree as ET
from pathlib import Path
import sys

try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environemnt variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')

root = sys.argv[1]
if len(sys.argv) > 2:
    generate_xml =  sys.argv[2]
else:
    generate_xml = "0"
prototxt = root+'/MobileNetSSD_deploy.prototxt'
weights = root+'/MobileNetSSD_deploy.caffemodel'

# Functions
def initialize_box_xml(image):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = image.split("/")[-2:-1][0]
    ET.SubElement(annotation, "filename").text = image[:-4]+"_crop.jpg"
    ET.SubElement(annotation, "path").text = image
    
    src = ET.SubElement(annotation, "source")
    ET.SubElement(src, "database").text = 'Unknown'
    
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(cols)
    ET.SubElement(size, "height").text = str(rows)
    ET.SubElement(size, "depth").text = "3"
    
    ET.SubElement(annotation, "segmented").text = "0"
    return annotation

# Initialize variables
inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5
num_classes = 20 ####Pour les animaux, 20 classes suffisent car il y a tout dedans
thr = 0.2

if num_classes == 20:
    net = cv.dnn.readNetFromCaffe(prototxt, weights)
    swapRB = False
    classNames = { 0: 'background',
        1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
        5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
        10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
        14: 'motorbike', 15: 'person', 16: 'pottedplant',
        17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
else:
    assert(num_classes == 90)
    net = cv.dnn.readNetFromTensorflow(weights, prototxt)
    swapRB = True
    classNames = { 0: 'background',
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
        7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
        13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
        18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
        24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
        32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
        37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
        41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
        46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
        67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
        75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
        86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }


# For each image in /image
pathlist = Path(root+'/image').glob('**/*.jpg')
for file in pathlist:
    # because path is object not string
    #file="D:\workspaces\mobilenet\image\43.jpg"
    image = str(file).replace("\\","/")
    img_name = image.split("/")[-1:][0]

    # Load image
    frame = cv.imread(image);
    # Convert for network
    blob = cv.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)
    net.setInput(blob)
    detections = net.forward()
    # Resize
    cols = frame.shape[1]
    rows = frame.shape[0]
    if cols / float(rows) > WHRatio:
        cropSize = (int(rows * WHRatio), rows)
    else:
        cropSize = (cols, int(cols / WHRatio))
    
    y1 = int((rows - cropSize[1]) / 2)
    y2 = y1 + cropSize[1]
    x1 = int((cols - cropSize[0]) / 2)
    x2 = x1 + cropSize[0]
    frame = frame[y1:y2, x1:x2]
    
    cols = frame.shape[1]
    rows = frame.shape[0]
    
    # Save cropped image
    if generate_xml is not None and generate_xml == "xml":
        cv.imwrite(root+"/generated/crop_"+img_name, frame)
    annotation = initialize_box_xml(image)
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > thr:
            class_id = int(detections[0, 0, i, 1])
    
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
    
            cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                
                # XML completion
                obj = ET.SubElement(annotation, "object")
                ET.SubElement(obj, "name").text = label
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                
                box = ET.SubElement(obj, "bndbox")
                ET.SubElement(box, "xmin").text = str(xLeftBottom)
                ET.SubElement(box, "ymin").text = str(yLeftBottom)
                ET.SubElement(box, "xmax").text = str(xRightTop)
                ET.SubElement(box, "ymax").text = str(yRightTop)
                
    # Save image with labels
    cv.imwrite(root+"/generated/labels_"+img_name, frame)
    
    # Save XML
    tree = ET.ElementTree(annotation)
    if generate_xml is not None and generate_xml == "xml":
        tree.write(root+"/generated/crop_"+img_name.split(".")[:-1][0]+".xml")
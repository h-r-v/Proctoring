import cv2
import numpy as np
from landmark_detector import get_square_box

def bound_box(box, w, h):
    
    x,y,x1,y1 = box

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x1 > w:
        x1 = w
    if y1 > h:
        y1 = h

    box = [x,y,x1,y1]
    return box

def get_face_detector(modelFile=None,
                      configFile=None,
                      quantized=False):
    """
    Get the face detection caffe model of OpenCV's DNN module
    
    Parameters
    ----------
    modelFile : string, optional
        Path to model file. The default is "models/res10_300x300_ssd_iter_140000.caffemodel" or models/opencv_face_detector_uint8.pb" based on quantization.
    configFile : string, optional
        Path to config file. The default is "models/deploy.prototxt" or "models/opencv_face_detector.pbtxt" based on quantization.
    quantization: bool, optional
        Determines whether to use quantized tf model or unquantized caffe model. The default is False.
    
    Returns
    -------
    model : dnn_Net
    """
    if quantized:
        if modelFile == None:
            modelFile = "models/opencv_face_detector_uint8.pb"
        if configFile == None:
            configFile = "models/opencv_face_detector.pbtxt"
        model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        
    else:
        if modelFile == None:
            modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        if configFile == None:
            configFile = "models/deploy.prototxt"
        model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model
    
def find_faces(img, model):
    """
    Find the faces in an image
    
    Parameters
    ----------
    img : np.uint8
        Image to find faces from
    model : dnn_Net
        Face detection model
    Returns
    -------
    faces : list
        List of coordinates of the faces detected in the image
    """
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            
            x, y, x1, y1 = get_square_box([x, y, x1, y1])
            x, y, x1, y1 = bound_box([x, y, x1, y1], w, h)
            
            faces.append([x, y, x1, y1])
    return faces
    
def draw_boxes(img, box_coordinates):
    """
    Draw faces on image
    Parameters
    ----------
    img : np.uint8
        Image to draw faces on
    faces : List of face coordinates
        Coordinates of faces to draw
    Returns
    -------
    None.
    """
    for x, y, x1, y1 in box_coordinates:
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 3)
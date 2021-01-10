from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2

def get_square_box(box):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        return box
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]

def get_landmark_model(saved_model='models/pose_model'):
    """
    Get the facial landmark model. 
    Original repository: https://github.com/yinguobing/cnn-facial-landmark
    Parameters
    ----------
    saved_model : string, optional
        Path to facial landmarks model. The default is 'models/pose_model'.
    Returns
    -------
    model : Tensorflow model
        Facial landmarks model
    """
    model = keras.models.load_model(saved_model)
    return model

def get_marks(frame, face, model):

    face_img = frame[face[1]: face[3],face[0]: face[2]]
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    landmark_prediction = model.signatures["predict"](tf.constant([face_img], dtype=tf.uint8))
    marks = np.array(landmark_prediction['output']).flatten()[:136]
    marks = np.reshape(marks, (-1,2))

    return marks

def draw_marks(image, facebox, marks, color=(0, 255, 0)):
    """
    Draw the facial landmarks on an image
    Parameters
    ----------
    image : np.uint8
        Image on which landmarks are to be drawn.
    marks : list or numpy array
        Facial landmark points
    color : tuple, optional
        Color to which landmarks are to be drawn with. The default is (0, 255, 0).
    Returns
    -------
    None.
    """

    marks *= (facebox[2] - facebox[0])
    marks[:, 0] += facebox[0]
    marks[:, 1] += facebox[1]

    for mark in marks:
        cv2.circle(image, (mark[0], mark[1]), 2, color, -1, cv2.LINE_AA)
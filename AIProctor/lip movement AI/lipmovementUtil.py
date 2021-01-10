from face_detector import get_face_detector, find_faces, draw_boxes
from landmark_detector import get_square_box, get_landmark_model, get_marks, draw_marks
import cv2
import dlib

detector_tf = get_face_detector()
landmark_model_tf = get_landmark_model('models/pose_model')

detector_dlib = dlib.get_frontal_face_detector()
landmark_model_dlib = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

innerupperlip = [61,62,63]
innerlowerlip = [65,66,67]
roi = innerupperlip+innerlowerlip

global dlip_dlib_avg, dlip_tf_avg
dlip_tf_avg=0.7
dlip_dlib_avg=0.7

assert len(innerlowerlip)==len(innerupperlip)

def mean(a):
    ans = 0
    for i in a:
        ans += (i/len(a))
    return ans

def ema(value, ema_prev, days=10, smoothing=2):
    #high smoothing = more weightage to new data
    ema_new = (value*(smoothing/(1+days))) + (ema_prev*(1-(smoothing/(1+days))))
    return ema_new

def lipmovementTF(frame, threshold=0.09):

    dlips_tf=[0.0]*len(innerupperlip)

    frame_tf = frame.copy()

    faces_tf = find_faces(frame_tf, detector_tf)

    draw_boxes(frame_tf, faces_tf)

    #TF
    for i, face in enumerate(faces_tf):

        if face == None:
            break

        face_img = frame_tf[face[1]: face[3],face[0]: face[2]].copy()
        landmarks_tf = get_marks(frame_tf, face, landmark_model_tf)
        
        #draw landmarks
        draw_marks(frame_tf, face, landmarks_tf[roi])

        for i in range(len(innerupperlip)):
            x = landmarks_tf[innerupperlip[i]][0]
            x1 = landmarks_tf[innerlowerlip[i]][0]
            y = landmarks_tf[innerupperlip[i]][1]
            y1 = landmarks_tf[innerlowerlip[i]][1]

            dlips_tf[i] = (abs(x1-x)+abs(y1-y))
        
        curr_mean = mean(dlips_tf)
        if curr_mean>0:     
            global dlip_tf_avg
            dlip_tf_avg = ema( curr_mean, dlip_tf_avg) 
        
        cv2.putText(frame_tf, "{:.2}".format(dlip_tf_avg), (face[0], face[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

        break

    #print(dlip_tf_avg)
    if dlip_tf_avg >= threshold:
        return (frame_tf,True)
    
    return (frame_tf,False)


def lipmovementDlib(frame, threshold=0.09):

    dlips_dlib=[0.0]*len(innerupperlip)

    frame_dlib = frame.copy()
    gray_frame = cv2.cvtColor(frame_dlib.copy(), cv2.COLOR_BGR2GRAY)

    faces_dlib = detector_dlib(gray_frame)

    #draw boxes
    temp_faces_dlib=[]
    for face in faces_dlib:
        x1 = face.left() 
        y1 = face.top() 
        x2 = face.right() 
        y2 = face.bottom()
        temp_faces_dlib.append([x1,y1,x2,y2])
    draw_boxes(frame_dlib, temp_faces_dlib)

    for face in faces_dlib: 

        x1 = face.left() 
        y1 = face.top() 
        x2 = face.right() 
        y2 = face.bottom() 

        w = abs(x2-x1)

        landmarks_dlib = landmark_model_dlib(gray_frame, face)

        #draw landmarks
        for n in roi: 
            x = landmarks_dlib.part(n).x 
            y = landmarks_dlib.part(n).y 

            cv2.circle(frame_dlib, (x, y), 2, (255, 255, 0), -1) 

        for i in range(len(innerupperlip)):
            x = landmarks_dlib.part(innerupperlip[i]).x
            x_end = landmarks_dlib.part(innerlowerlip[i]).x 
            y = landmarks_dlib.part(innerupperlip[i]).y
            y_end = landmarks_dlib.part(innerlowerlip[i]).y

            dlips_dlib[i] = (abs(x_end-x)+abs(y_end-y))/w

        curr_mean = mean(dlips_dlib)
        if curr_mean>0:     
            global dlip_dlib_avg
            dlip_dlib_avg = ema( curr_mean, dlip_dlib_avg) 
        cv2.putText(frame_dlib, "{:.2}".format(dlip_dlib_avg), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

        break

    if dlip_dlib_avg >= threshold:
        return (frame_dlib,True)
    
    return (frame_dlib,False)


if __name__=='__main__':
    vid = cv2.VideoCapture(0) 

    while(True):

        ret, frame = vid.read()

        FrameTF, ansTF = lipmovementTF(frame)
        FrameDlib, ansDlib = lipmovementDlib(frame, 0.07)
        
        cv2.imshow("TF", FrameTF)
        cv2.imshow("Dlib", FrameDlib)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    vid.release() 
    cv2.destroyAllWindows() 
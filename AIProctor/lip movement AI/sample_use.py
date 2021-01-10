from lipmovementUtil import lipmovementTF, lipmovementDlib
import cv2

vid = cv2.VideoCapture(0) 

while(True):

    ret, frame = vid.read()

    FrameTF, ansTF = lipmovementTF(frame)
    FrameDlib, ansDlib = lipmovementDlib(frame)
    
    if ansDlib:
        cv2.putText(frame, "TALKING", (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

    cv2.imshow('Preview', frame)
    cv2.imshow('Dlib', FrameDlib)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release() 
cv2.destroyAllWindows() 
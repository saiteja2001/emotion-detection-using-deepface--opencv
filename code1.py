import cv2
from deepface import DeepFace
import time
cap=cv2.VideoCapture("video_1.mp4")
output = cv2.VideoWriter('output1.mp4',cv2.VideoWriter_fourcc(*'mp4v'),10,(700,500))
print("Playing Video.....")
while cap.isOpened():
    _,frame=cap.read()
    if _ == True:
        try:
            predictions = DeepFace.analyze(frame, actions=['emotion'])
            emotion = predictions['dominant_emotion']
            cv2.putText(frame,emotion,(5,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            x,y,w,h=int(predictions['region']['x']),int(predictions['region']['y']),int(predictions['region']['w']),int(predictions['region']['h'])
            cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 1)
        except:
            print("face not found")
        frame = cv2.resize(frame, (700, 500))
        cv2.imshow("video", frame)
        output.write(frame)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
    else:
        break

cap.release()
output.release()
cv2.destroyAllWindows()

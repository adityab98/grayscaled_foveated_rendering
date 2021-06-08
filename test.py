import numpy as np
import cv2
import ffmpeg
import skvideo.io

#print(skvideo.__file__)

cap = cv2.VideoCapture("file.mp4")
output='output.avi'
width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#codec=cv2.VideoWriter_fourcc('F','M','P','4')
#vid_writer=cv2.VideoWriter(output,cv2.CAP_FFMPEG,codec,24,(480,640),0)
vid_writer=skvideo.io.FFmpegWriter(output,outputdict={
    '-vcodec':'libx264','-b':'1290000'
    })
while(cap.isOpened()):
    ret, frame = cap.read()
    #if ret==True:
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vid_writer.writeFrame(gray)
    #cv2.imshow("Camera frame", frame)
    #cv2.waitKey(1)
    #cv2.imshow('frame',gray)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
cap.release()
vid_writer.close()
#vid_writer.release()
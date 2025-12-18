from FaceMeshModule import FaceMeshDetection
import cv2
import mediapipe as mp
import time
import math
import numpy as np

detector=FaceMeshDetection()
cap=cv2.VideoCapture(0)
ptime=0
def abs_euclid_dist(point1,point2):
    point1=np.array(point1) 
    point2=np.array(point2)
    distance = np.linalg.norm(point1 - point2)
    return distance
window_time=10
blink_history=[]
ear_history=[]
mar_history=[]
while True:
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    ret,img=cap.read()
    img,faces=detector.findFaceMesh(img,False)
    if len(faces)!=0:
        face=faces[0]
        #points for EAR
        #left eye
        left_eye_out_corner=face[33]
        left_up_eye_in=face[160]
        left_up_eye_out=face[158]
        left_eye_in_corner=face[133]
        left_low_eye_out=face[153]
        left_low_eye_in=face[144]
        #right eye
        right_eye_out_corner=face[362]
        right_up_eye_in=face[385]
        right_up_eye_out=face[387]
        right_eye_in_corner=face[263]
        right_low_eye_out=face[373]
        right_low_eye_in=face[380]

    eye_points=[face[33],face[160],face[158],face[133],face[153],face[144],face[362],face[385],face[387],face[263],face[373],face[380]]
    for i in eye_points:
        cv2.circle(img,i,2,(255,0,255),-1)
    

    EAR_left=(abs_euclid_dist(left_up_eye_in,left_low_eye_in)+abs_euclid_dist(left_up_eye_out,left_low_eye_out))/(2*abs_euclid_dist(left_eye_out_corner,left_eye_in_corner))
    EAR_right=(abs_euclid_dist(right_up_eye_in,right_low_eye_in)+abs_euclid_dist(right_up_eye_out,right_low_eye_out))/(2*abs_euclid_dist(right_eye_out_corner,right_eye_in_corner))
    
    EAR_mean = (EAR_left + EAR_right) / 2
    ear_history.append((time.time(), EAR_mean))
    cv2.putText(img,f'EAR MEAN:{float(EAR_mean):.3f}',(20,90),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    cv2.putText(img,f'EAR LEFT:{float(EAR_left):.3f}',(20,120),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    cv2.putText(img,f'EAR RIGHT:{float(EAR_right):.3f}',(20,150),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    #poiints for mouth MAR
    mouth_left_corner=face[61]
    mouth_right_corner=face[291]
    mouth_left_up_lip=face[78]
    mouth_up_lip=face[13]
    mouth_right_low_lip=face[308]
    mouth_low_lip=face[14]
    

    mouth_points=[face[61],face[291],face[78],face[13],face[308],face[14]]
    for i in mouth_points:
        cv2.circle(img,i,2,(255,0,255),-1)
    MAR=(abs_euclid_dist(mouth_up_lip,mouth_low_lip)+abs_euclid_dist(mouth_left_up_lip,mouth_right_low_lip))/(2*abs_euclid_dist(mouth_left_corner,mouth_right_corner))
    mar_history.append((time.time(), MAR))

    cv2.putText(img,f'MAR:{float(MAR):.3f}',(20,180),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    cv2.imshow("capture",img)

    # window updates
    #ear update
    now=time.time()
    temp=[]
    for(i,j) in ear_history:
        if now - i <= window_time:
            temp.append((i, j))
    ear_history=temp
    #mar update
    temp=[]
    for(i,j) in mar_history:
        if now - i <= window_time:
            temp.append((i, j))
    mar_history=temp
    #blikn update
    temp=[]
    for(i,j) in blink_history:
        if now - i <= window_time:
            temp.append((i, j))
    blink_history=temp

    # value extraction
    ear_values=[]
    mar_values=[]
    blink_times=[]
    for (i,j) in ear_history:
        ear_values.append(j)

    for (i,j) in mar_history:
        mar_values.append(j)
    for (i,j) in blink_history:
        blink_times.append(j)
    #computing values
    blink_count_10s = len(blink_times)
    avg_blink_duration_10s = sum(blink_times)/blink_count_10s
    EAR_min = min(ear_values)
    EAR_var = np.var(ear_values)
    MAR_mean = np.mean(mar_values)
    MAR_var  = np.var(mar_values) 



    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
from FaceMeshModule import FaceMeshDetection
import cv2
import mediapipe as mp
import time
import math
import numpy as np
def rotationMatrixToEulerAngles(R):
    roll = math.atan2(R[1,0],R[0,0])
    pitch = math.atan2(-R[2,0],math.sqrt(R[2,1]**2 + R[2,2]**2))
    yaw = math.atan2(R[2,1],R[2,2])


    return (
        math.degrees(pitch),
        math.degrees(yaw),
        math.degrees(roll)
    )
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
blink_state=0
blink_start_time=0
BASELINE_WINDOW = 25
blink_ratio_thresh=0.75
ear_baseline_history = []
mar_baseline_history = []
ear_ratio_history=[]
mar_ratio_history=[]
# 3d refrence model for pnp(in mm)(nose tip,chin,left eye,right eye,left mouth corner,right mouth corner)
refrence_3d_face=np.array([(0.0,0.0,0.0),(0,-63.6,-12.0),(-45.0,17.0,-20.0),(45.0,17.0,-20.0),(-30.0,-50.0,-12.0),(30.0,-50.0,-12.0)],dtype=np.float64)

while True:
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    ret,img=cap.read()
    #intrinsic camera matrix for pnp 
    h,w=img.shape[:2]
    focal_length=w
    camera_intrisic_matrix=np.array([[focal_length,0,w/2],[0,focal_length,h/2],[0,0,1]],dtype=np.float64)
    distortion_coefficients=np.zeros((4,1))# assuming no distortion

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
        

        cv2.putText(img,f'MAR:{float(MAR):.3f}',(20,180),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
        cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
        
        #ear znd mar normalization
        now = time.time()
        ear_baseline_history.append((now, EAR_mean))
        mar_baseline_history.append((now, MAR))
        temp = []
        for (t, v) in ear_baseline_history:
            if now - t <= BASELINE_WINDOW:
                temp.append((t, v))
        ear_baseline_history = temp
        temp = []
        for (t, v) in mar_baseline_history:
            if now - t <= BASELINE_WINDOW:
                temp.append((t, v))
        mar_baseline_history = temp
        
        if len(ear_baseline_history) > 5:
            ear_baseline = np.mean([v for (_, v) in ear_baseline_history])
        else:
            ear_baseline = EAR_mean  # fallback

        if len(mar_baseline_history) > 5:
            mar_baseline = np.mean([v for (_, v) in mar_baseline_history])
        else:
            mar_baseline = MAR
        EAR_ratio = EAR_mean / (ear_baseline + 1e-6)
        MAR_ratio = MAR / (mar_baseline + 1e-6)
        
        ear_ratio_history.append((now, EAR_ratio))
        mar_ratio_history.append((now, MAR_ratio))
        #blink logic using ratios
        
        now=time.time()
        if EAR_ratio<blink_ratio_thresh and blink_state==0:
            blink_state=1
            blink_start_time=now
        elif EAR_ratio>=blink_ratio_thresh and blink_state==1:
            blink_state=0
            blink_end_time=now
            duration=blink_end_time-blink_start_time
            blink_history.append((blink_end_time, duration))
        # window updates
        #ear update
        now=time.time()
        if len(ear_ratio_history)!=0:
            temp=[]
            for(i,j) in ear_ratio_history:
                if now - i <= window_time:
                    temp.append((i, j))
            ear_ratio_history=temp
        else:
            print("please make sure your face is visible")
        #mar update
        if len(mar_ratio_history)!=0:
            temp=[]
            for(i,j) in mar_ratio_history:
                if now - i <= window_time:
                    temp.append((i, j))
            mar_ratio_history=temp
        else:
            print("please make sure your face is visible")
        #blikn update
        if len(blink_history)!=0:
            temp=[]
            for(i,j) in blink_history:
                if now - i <= window_time:
                    temp.append((i, j))
            blink_history=temp
        else:
            print("please make sure your face is visible")
        #pnp logic
        #same format as refrence 3d points(pnp)
        
        face_2d_points = np.array([
        face[1],     # Nose tip
        face[152],   # Chin
        face[33],    # Left eye
        face[263],   # Right eye
        face[61],    # Left mouth
        face[291]    # Right mouth
        ], dtype=np.float64)
        success,rotation_vector,translation_vector=cv2.solvePnP(refrence_3d_face,face_2d_points,camera_intrisic_matrix,distortion_coefficients,flags=cv2.SOLVEPNP_ITERATIVE)
        p=0.0
        r=0.0
        y=0.0
        if success:
            rotation_matrix_cp,_=cv2.Rodrigues(rotation_vector)#camera perspectivve rot. matrix
            camp_to_worldp = np.array([
            [ 0,  0,  1],   # X_world ← Z_cam
            [-1,  0,  0],   # Y_world ← -X_cam
            [ 0, -1,  0]    # Z_world ← -Y_cam
            ], dtype=np.float64)
            rotation_matrix_wp=rotation_matrix_cp@camp_to_worldp# rot. matrix world perspective
            p,y,r=rotationMatrixToEulerAngles(rotation_matrix_wp)#pitch,roll,yaw
        cv2.putText(img,f"pitch:{p:.2f}",(20,210),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(img,f"yaw:{y:.2f}",(20,240),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(img,f"roll:{r:.2f}",(20,270),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow("capture",img)
        # value extraction
        ear_values=[]
        mar_values=[]
        blink_times=[]

        for (i,j) in ear_ratio_history:
            ear_values.append(j)

        for (i,j) in mar_ratio_history:
            mar_values.append(j)
        for (i,j) in blink_history:
            blink_times.append(j)
        #computing values
        blink_count_10s = len(blink_times)
        if blink_count_10s>0:
            avg_blink_duration_10s = sum(blink_times)/blink_count_10s
        else:
            avg_blink_duration_10s=0
        if len(ear_values)>0:

            EAR_min = min(ear_values)
            
        else:
            EAR_min= 0
            
        if len(mar_values)>0:
            MAR_mean = np.mean(mar_values)
             
        else:
            MAR_mean=0
            
        if len(mar_values)>1:
            MAR_var  = np.var(mar_values)
        else:
            MAR_var=0
        if len(ear_values)>1:
            EAR_var  = np.var(ear_values)
        else:
            EAR_var=0


    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
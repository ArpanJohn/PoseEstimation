#getting the data and running the model
import numpy as np
import cv2
import msgpack as msgp
import msgpack_numpy as mpn
import glob
import os
import time
import mediapipe as mp
from support.funcs import *
import pandas as pd

# Setting the parameters of the stream
h=480 #720 
w=640 #1280
fps=30
windowscale=0.6

def find_midpoint(point1, point2):
    midpoint = []
    for i in range(len(point1)):
        coord_avg = (point1[i] + point2[i]) / 2.0
        midpoint.append(coord_avg)
    return midpoint

# Initializing the model to locate the landmarks
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initializing the landmark lists
LH, LK, LA, LT, RH, RK, RA, RT = [],[],[],[],[],[],[],[]
M=[]
 
# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
 
# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

frames=0

pth = r"C:\Users\arpan\OneDrive\Documents\internship\rec_program\savdir\Session 16-06-23_15-59-27_631"

lst = os.listdir(pth)
vid_name = lst[-1]

targetPattern = f"{pth}\\DEPTH*"
campth = glob.glob(targetPattern)

targetPattern_param = f"{pth}\\PARAM*"
ppth = glob.glob(targetPattern_param)

targetPattern_colour = f"{pth}\\COLOUR*"
cpth = glob.glob(targetPattern_colour)

# campth.pop(0)
# cpth.pop(0)

print(campth)
print(cpth)
print(ppth)

img = []
c=0
for i in cpth:
    print(i)
    col_file = open(i, "rb")
    unpacker = None
    unpacker = msgp.Unpacker(col_file, object_hook=mpn.decode)
    for unpacked in unpacker:
        c+=1
        unpacked=cv2.flip(unpacked,1)
        imagep=unpacked

        # Making predictions using holistic model
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        imagep.flags.writeable = False
        results = holistic_model.process(imagep)
        imagep.flags.writeable = True

        # Converting back the RGB image to BGR
        color_image = imagep

        #Drawing the pose landmarks
        mp_drawing.draw_landmarks(
        imagep,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS)

        # Calculating the FPS
        currentTime = time.time()
        fps = 1 / (currentTime-previousTime)
        previousTime = currentTime

        # Displaying FPS on the image
        cv2.putText(imagep, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.putText(imagep, str(int(frames))+' total_frames', (500, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        frames+=1

        # Finding and saving the landmark positions        
        dic = {}
        for mark, data_point in zip(mp_holistic.PoseLandmark, results.pose_landmarks.landmark):
            dic[mark.value] = dict(landmark = mark.name, 
                x = data_point.x,
                y = data_point.y)        
        try:
            LH.append([dic[23]['x']*w,dic[23]['y']*h])
        except:
            LH.append(np.nan)
        try:
            LK.append([dic[25]['x']*w,dic[25]['y']*h])
        except:
            LK.append(np.nan)
        try:
            LA.append([dic[27]['x']*w,dic[27]['y']*h])
        except:
            LA.append(np.nan)
        try:
            LT.append([dic[31]['x']*w,dic[31]['y']*h])
        except:
            LT.append(np.nan)
        try:
            RH.append([dic[24]['x']*w,dic[24]['y']*h])
        except:
            RH.append(np.nan)
        try:
            RK.append([dic[26]['x']*w,dic[26]['y']*h])
        except:
            RK.append(np.nan)
        try:
            RA.append([dic[28]['x']*w,dic[28]['y']*h])
        except:
            RA.append(np.nan)
        try:
            RT.append([dic[32]['x']*w,dic[32]['y']*h])
        except:
            RT.append(np.nan)
        try:
            Smid=midpoint([dic[23]['x']*w,dic[23]['y']*h],[dic[24]['x']*w,dic[24]['y']*h])
            perpx=int(Smid[0])
            perpy=(int(Smid[1])-25)

            cv2.circle(color_image,(perpx,perpy) , 5, (0, 0, 255), 2)
            M.append([perpx,perpy])     #in uv format  
        except:
            M.append(np.nan)
        try:
            draw_box(color_image,LH[c],LK[c])
            draw_box(color_image,LK[c],LA[c])
            draw_box(color_image,LA[c],LT[c])
            
            draw_box(color_image,RH[c],RK[c])
            draw_box(color_image,RK[c],RA[c])
            draw_box(color_image,RA[c],RT[c])
        except:
            pass

        # Display the resulting image
        cv2.imshow("Pose Landmarks", imagep)

        # Enter key 'q' to break the loopqq
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
         
        img.append(unpacked)
        #cv2.imshow("sadf", unpacked)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        try:
            if (unpacked)==-1:
                cv2.destroyAllWindows()
                break
        except:
            continue
    col_file.close()

cv2.destroyAllWindows()

pos = []
c=0
for i in campth:
    print(i)
    depth_file = open(i, "rb")
    unpacker = None
    unpacker = msgp.Unpacker(depth_file, object_hook=mpn.decode)
    for unpacked in unpacker:
         c+=1

         pos.append(unpacked)
    depth_file.close()

cv2.destroyAllWindows()
pos=np.array(pos)
print(pos.shape)

#obtaining list time_stamps
p = open(ppth[0], "rb")
unpacker=None
unpacker = msgp.Unpacker(p, object_hook=mpn.decode)
prm = []
for unpacked in unpacker:
    prm.append(unpacked)

timestamps=prm

land_marks={'LH':LH,'LK':LK,'LA':LA,'LT':LT,'RH':RH,'RK':RK,'RA':RA,'RT':RT}

pos=np.array(pos)
print(pos.shape)
df=pd.DataFrame()
xyz=['_x','_y','_z']

df['epoch_time']=pd.Series(timestamps)

for key,value in land_marks.items():    
    for j in range(3):
        data=[]
        for i in range(len(pos)):
            try:
                x=pos[i][int((value[i][1]))][int((value[i][0]))][j]
                data.append(x)
            except:
                continue
        df[key+xyz[j]]=pd.Series(data)

df.to_csv('mpipe.csv', index=False)

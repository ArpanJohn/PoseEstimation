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
from natsort import natsorted
import re
import json

# Read the JSON file containing the Session Directory
with open('upperbody\SessionDirectory.json', 'r') as file:
    session_data = json.load(file)

# Get the directory path from the JSON data
pth = session_data["directory"]

lst = os.listdir(pth)

targetPattern = f"{pth}\\POINT*"
campth = glob.glob(targetPattern)

targetPattern_param = f"{pth}\\PARAM*"
ppth = glob.glob(targetPattern_param)

targetPattern_colour = f"{pth}\\COLOUR*"
cpth = glob.glob(targetPattern_colour)

#obtaining parameters and list time_stamps
p = open(ppth[0], "rb")
unpacker=None
unpacker = list(msgp.Unpacker(p, object_hook=mpn.decode))
timestamps = []
ps = []

# Getting the parameters of the recording
parameters=unpacker.pop(0)

# removing formating things
parameters=parameters.replace('x', ' ')
parameters=parameters.replace(':', ' ')
parameters=parameters.replace(']', '')
parameters=parameters.replace('[', '')

# removing letters
modified_string = re.sub('[a-zA-Z]', '', parameters)
modified_string = modified_string.strip()

# splitting the string and assigning the parameters
params = modified_string.split(' ')

w = int(params[0])
h = int(params[1])
fps = int(params[-1])

for unpacked in unpacker:
    timestamps.append(unpacked)

rec_dur=timestamps[-1]-timestamps[0]

# Print the parameters of the recording
print(('recording duration '+f"{rec_dur:.3}"+' s'+'\nresolution :'+str(w)+'x'+str(h)+ '; fps : '+str(fps)))
print('number of frames:', len(timestamps))

# Sorting the color and depth files
cpth=natsorted(cpth)
campth=natsorted(campth)

# Initializing the landmark lists
LS, LE, LW, RS, RE, RW, TR = [], [], [], [], [], [], []
RI,LI=[],[]
 
# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
 
# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

frames=0

c=0

# Initializing the model to locate the landmarks
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

for i in cpth:
    # print(i)
    col_file = open(i, "rb")
    unpacker = None
    unpacker = msgp.Unpacker(col_file, object_hook=mpn.decode)
    for unpacked in unpacker:
        c+=1
        imagep=unpacked
        
        # Making predictions using holistic model
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        imagep.flags.writeable = False
        results = holistic_model.process(imagep)
        try:
            imagep.flags.writeable = True
        except:
            imagep.flags.writeable = False

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
        cv2.putText(color_image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.putText(color_image, str(int(frames))+' total_frames', (900, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.putText(color_image, str(i.split('\\')[-1]), (10, 650), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
        frames+=1

        # Finding and saving the landmark positions        
        try:
            dic = {}
            for mark, data_point in zip(mp_holistic.PoseLandmark, results.pose_landmarks.landmark):
                dic[mark.value] = dict(landmark = mark.name, 
                    x = data_point.x,
                    y = data_point.y)       
            
            try:
                Smid=midpoint([dic[11]['x']*w,dic[11]['y']*h],[dic[12]['x']*w,dic[12]['y']*h])
                perpx=int(Smid[0])
                perpy=(int(Smid[1])+25)

                cv2.circle(color_image,(perpx,perpy) , 5, (0, 0, 255), 2)

            except:
                pass

            try:
                # Drawing the boxes around limbs for occlusion
                draw_box(color_image,[dic[11]['x']*w,dic[11]['y']*h],[dic[13]['x']*w,dic[13]['y']*h])
                draw_box(color_image,[dic[12]['x']*w,dic[12]['y']*h],[dic[14]['x']*w,dic[14]['y']*h])
                draw_box(color_image,[dic[13]['x']*w,dic[13]['y']*h],[dic[15]['x']*w,dic[15]['y']*h])
                draw_box(color_image,[dic[14]['x']*w,dic[14]['y']*h],[dic[16]['x']*w,dic[16]['y']*h])
                draw_box(color_image,[dic[16]['x']*w,dic[16]['y']*h],([dic[20]['x']*w,dic[20]['y']*h]),(255,0,255),40)
                draw_box(color_image,[dic[15]['x']*w,dic[15]['y']*h],([dic[19]['x']*w,dic[19]['y']*h]),(0,255,255),40) 
            except:
                pass
        except:
            pass

        # Enter key 'q' to break the loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
         
        cv2.imshow("pose landmarks", color_image)
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
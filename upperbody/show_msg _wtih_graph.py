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
from natsort import natsorted
import re as re_str
import json
import time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline, interp1d
import math

# Measure the execution time
start_time = time.time()

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

try:
    # getting mpipe data
    df_mpipe=pd.read_csv(pth+'\\mpipe_tasks.csv')

    # getting mocap data
    df_mocap=pd.read_csv(pth+'\\mocap_tasks.csv')
except:
    print('run the Euler angle notebook')
    quit()

# Calculating the elbow angle using mediapipe and mocap data
mpipeRightElbowAngle, mocapRightElbowAngle = [], []
mpipeLeftElbowAngle, mocapLeftElbowAngle = [], []

tasks_data=[]
# Calculate right elbow angle using mediapipe data
for i in range(len(df_mpipe['epoch_time'].tolist())):
    
    mpipeRightElbowAngle.append(angle3point([df_mpipe['RS_x'][i], df_mpipe['RS_y'][i], df_mpipe['RS_z'][i]],
                                            [df_mpipe['RE_x'][i], df_mpipe['RE_y'][i], df_mpipe['RE_z'][i]],
                                            [df_mpipe['RW_x'][i], df_mpipe['RW_y'][i], df_mpipe['RW_z'][i]]))

    mpipeLeftElbowAngle.append(angle3point([df_mpipe['LS_x'][i], df_mpipe['LS_y'][i], df_mpipe['LS_z'][i]],
                                           [df_mpipe['LE_x'][i], df_mpipe['LE_y'][i], df_mpipe['LE_z'][i]],
                                           [df_mpipe['LW_x'][i], df_mpipe['LW_y'][i], df_mpipe['LW_z'][i]]))

# Calculate right elbow angle using mocap data
for i in range(len(df_mocap['epoch_time'].to_numpy())):
    mocapRightElbowAngle.append(angle3point([df_mocap['RS_x'][i], df_mocap['RS_y'][i], df_mocap['RS_z'][i]],
                                            [df_mocap['RE_x'][i], df_mocap['RE_y'][i], df_mocap['RE_z'][i]],
                                            [df_mocap['RW_x'][i], df_mocap['RW_y'][i], df_mocap['RW_z'][i]]))

    mocapLeftElbowAngle.append(angle3point([df_mocap['LS_x'][i], df_mocap['LS_y'][i], df_mocap['LS_z'][i]],
                                           [df_mocap['LE_x'][i], df_mocap['LE_y'][i], df_mocap['LE_z'][i]],
                                           [df_mocap['LW_x'][i], df_mocap['LW_y'][i], df_mocap['LW_z'][i]]))

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
modified_string = re_str.sub('[a-zA-Z]', '', parameters)
modified_string = modified_string.strip()

# splitting the string and assigning the parameters
params = modified_string.split(' ')

w = int(params[0])
h = int(params[1])
fps = int(params[-1])

for unpacked in unpacker:
    timestamps.append(unpacked)

rec_dur=timestamps[-1]-timestamps[0]

try:
    # Read the JSON file and retrieve the dictionary
    filename = pth+"\\task_markers.json"
    with open(filename, 'r') as file:
        task_markers = json.load(file)
    task_marker=1
except:
    print('no task markers')

# Print the parameters of the recording
print(('recording duration '+f"{rec_dur:.3}"+' s'+'\nresolution :'+str(w)+'x'+str(h)+ '; fps : '+str(fps)))
print('number of frames:', len(timestamps))

# Sorting the color and depth files
cpth=natsorted(cpth)
campth=natsorted(campth)

# Initializing the landmark lists
LS, LE, LW, RS, RE, RW, TR = [],[],[],[],[],[],[]
RI,LI,RT,LT,LP,RP=[],[],[],[],[],[]
 
# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
 
# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0
frames=0

c=0

interval = math.ceil(len(df_mocap['epoch_time'].to_numpy())/len(timestamps))

epoch_time=df_mocap['epoch_time'].to_numpy()[0]
fig, ax = plt.subplots()
x_data1 = list(df_mpipe['epoch_time'].to_numpy()-epoch_time)[0:-1:interval]
y_data1 = mocapRightElbowAngle[0:-1:interval]
x_data2 = list(df_mpipe['epoch_time'].to_numpy()-epoch_time)[0:-1:interval]
y_data2 = mpipeRightElbowAngle[0:-1:interval]

# Initializing the model to locate the landmarks
mp_holistic = mp.solutions.pose
holistic_model = mp_holistic.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.9
)

size = (w, h)

# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
result = cv2.VideoWriter(pth+'\\video.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

for i,j in zip(cpth,campth):

    try:
        col_file = open(i, "rb")
        unpacker = None
        unpacker = msgp.Unpacker(col_file, object_hook=mpn.decode)
        depth_file = open(j, "rb")
        d_unpacker = None
        d_unpacker = msgp.Unpacker(depth_file, object_hook=mpn.decode)
        for unpacked,d_unpacked in zip(unpacker,d_unpacker):
            c+=1
            # define the contrast and brightness value
            contrast = 1.5 # Contrast control ( 0 to 127)
            brightness =20  # Brightness control (0-100)

            img = cv2.cvtColor(unpacked, cv2.COLOR_RGB2BGR)
            imagep=cv2.addWeighted( img, contrast, img, 0, brightness)
            imagep = cv2.cvtColor(imagep, cv2.COLOR_BGR2RGB)
            imagep=np.asanyarray(imagep)

            pointcloud=np.asanyarray(d_unpacked)

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

            color_image_save = color_image

            # Displaying FPS on the image
            cv2.putText(color_image, str(int(fps))+" FPS", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(color_image, str(int(frames))+'frames', (480, 70), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 2)
            cv2.putText(color_image_save, str(f"{timestamps[frames]-timestamps[0]:.2f}")+' sec', (480, 30), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 2)
            cv2.putText(color_image, str(i.split('\\')[-1]), (10, 460), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(color_image_save, str(i.split('\\')[-1]), (10, 460), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            try:
                cv2.putText(color_image_save, str('task'+str(task_marker)), (480, 460), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,0,0), 2)
                if timestamps[frames]-timestamps[0]>task_markers['task'+str(task_marker)]:
                    task_marker+=1
                    print('next task')
            except:
                pass

            # Clear the graph
            ax.cla()
            st=0
            if frames > 50:
                st=frames-50
            # Plot the data
            ax.plot(x_data1[st:frames],y_data1[st:frames],color='red')
            ax.plot(x_data2[st:frames],y_data2[st:frames],color='blue')

            plt.legend(['mocap Right Elbow angle','mpipe Right elbow angle'])

            # Adjust the plot limits if necessary
            ax.relim()
            ax.autoscale
            ax.set_ylim(0,180)
            ax.autoscale_view()

            # Update the plot
            plt.draw()
            plt.pause(0.001)
            
            frames+=1
            
            # Enter key 'q' to break the loop
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            
            cv2.imshow("pose landmarks", color_image)
            # Write the frame into the
            # file 'filename.avi'
            result.write(color_image_save)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            try:
                if (unpacked)==-1:
                    cv2.destroyAllWindows()
                    break
            except:
                continue
    except:
        continue

    depth_file.close()
    col_file.close()

result.release()
cv2.destroyAllWindows()
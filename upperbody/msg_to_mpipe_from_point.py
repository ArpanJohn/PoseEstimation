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

fig, ax = plt.subplots()
ax.set_ylim(-480,0)
ax.set_xlim(0,640)
x_data1 = []
y_data1 = []
x_data2 = []
y_data2 = []

# Initializing the model to locate the landmarks
mp_holistic = mp.solutions.pose
holistic_model = mp_holistic.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.9
)

# Dictionary containing landmark names and corresponding values with mediapipe landmark number
pose_land_marks = {'LS': [LS,11], 'LE': [LE,13], 'LW': [LW,15], 'RS': [RS,12], 'RE': [RE,14], 'RW': [RW,16], 'TR': [TR,0]}
l_hand_land_marks={'LI':[LI,5],'LT':[LT,2],'LP':[LP,17]}
r_hand_land_marks={'RI':[RI,5],'RT':[RT,2],'RP':[RP,17]}

# pandas dataframe to hold landmark values
df=pd.DataFrame()
xyz=['_x','_y','_z']

df['epoch_time']=pd.Series(timestamps)

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
            cv2.putText(color_image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            # cv2.putText(color_image, str(int(frames))+' total_frames', (400, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(color_image, str(i.split('\\')[-1]), (10, 460), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(color_image_save, str(i.split('\\')[-1]), (10, 460), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(color_image_save, str(f"{timestamps[frames]-timestamps[0]:.2f}")+' seconds', (400, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

            frames+=1

            # Finding and saving the landmark positions        
            try:
                dic = {}
                for mark, data_point in zip(mp_holistic.PoseLandmark, results.pose_landmarks.landmark):
                    dic[mark.value] = dict(landmark = mark.name, 
                        x = data_point.x,
                        y = data_point.y,
                        vis = data_point.visibility)  
                for key,value in pose_land_marks.items():    
                                if value[1] !=0:     
                                    try:
                                        value[0].append(pointcloud[int(dic[value[1]]['y']*h)][int(dic[value[1]]['x']*w)])
                                    except:
                                        value[0].append(np.array([np.nan,np.nan,np.nan]))
                                else:
                                    try:
                                        Smid=midpoint([dic[11]['x']*w,dic[11]['y']*h],[dic[12]['x']*w,dic[12]['y']*h])
                                        perpx=int(Smid[0])
                                        perpy=(int(Smid[1])+50)

                                        cv2.circle(color_image,(perpx,perpy) , 5, (0, 0, 255), 2)
                                        TR.append(pointcloud[perpy][perpx])     #in uv format  
                                    except:
                                        TR.append(np.array([np.nan,np.nan,np.nan]))

                try: # boxing
                    # Drawing the boxes around limbs for occlusion
                    lub=draw_box(color_image,[dic[11]['x']*w,dic[11]['y']*h],[dic[13]['x']*w,dic[13]['y']*h])
                    rub=draw_box(color_image,[dic[12]['x']*w,dic[12]['y']*h],[dic[14]['x']*w,dic[14]['y']*h])
                    llb=draw_box(color_image,[dic[13]['x']*w,dic[13]['y']*h],[dic[15]['x']*w,dic[15]['y']*h])
                    rlb=draw_box(color_image,[dic[14]['x']*w,dic[14]['y']*h],[dic[16]['x']*w,dic[16]['y']*h])
                    lhb=draw_box(color_image,[dic[15]['x']*w,dic[15]['y']*h],([dic[19]['x']*w,dic[19]['y']*h]),(0,0,255),dis = 35) 
                    rhb=draw_box(color_image,[dic[16]['x']*w,dic[16]['y']*h],([dic[20]['x']*w,dic[20]['y']*h]),(255,0,0),dis = 35)

                    box_dic = {
                        'lub': [lub, ['LS', 'LE']],
                        'rub': [rub, ['RS', 'RE']],
                        'llb': [llb, ['LE', 'LW']],
                        'rlb': [rlb, ['RE', 'RW']],
                        'lhb': [lhb, ['LW']],
                        'rhb': [rhb, ['RW']]
                    }

                    # Iterate through each landmark and associated box in the dictionary
                    for k, j in pose_land_marks.items():
                        for key, values in box_dic.items():
                            # Check if the landmark is inside the box
                            try:
                                if j[-1]!=0:
                                    if point_in_quad([dic[j[-1]]['x']*w,dic[j[-1]]['y']*h], values[0]) and k not in values[-1]:
                                        for p in values[-1]:
                                            if j[0][pd.Series(j[0][-1]).last_valid_index()][-1]+0.2 > pose_land_marks[p][0][pd.Series(pose_land_marks[p][0][-1]).last_valid_index()][-1] or np.isnan(j[0][-1]).any():
                                                # print(k, 'is occluded by', key, p, 'at frame', frames,str(f" \t{timestamps[frames]-timestamps[0]:.2f}")+' seconds')
                                                j[0][-1] = [np.nan,np.nan,np.nan]
                                                # print('corrected')
                                elif point_in_quad([perpx,perpy], values[0]) and k not in values[-1]:
                                    # print(k, 'is occluded by', key, 'at frame', frames, str(f" \t{timestamps[frames]-timestamps[0]:.2f}")+' seconds')
                                    j[0][-1] = [np.nan,np.nan,np.nan]
                                    # print('corrected')
                            except:
                                pass
                except:
                    pass

                # Clear the graph
                # ax.cla()

                # ax.set_ylim(-h,0)
                # ax.set_xlim(0,w)

                # Append the data to the lists
                # x_data1.append(dic[16]['x']*w)
                # y_data1.append(-dic[16]['y']*h)

                # x_data2.append(dic[12]['x']*w)
                # y_data2.append(-dic[12]['y']*h)

                # Plot the data
                # ax.plot(x_data1[-20:],y_data1[-20:],color='red')
                # ax.plot(x_data2[-20:],y_data2[-20:],color='blue')

                # plt.legend(['Right Wrist','Right Shoulder'])

                # Adjust the plot limits if necessary
                # ax.set_xlim(y_data1[-20],y_data1[-1])
                # ax.relim()
                # ax.autoscale_view()

                # Update the plot
                # plt.draw()
                # plt.pause(0.001)
            except:
                LS.append([np.nan,np.nan,np.nan])
                LE.append([np.nan,np.nan,np.nan])
                LW.append([np.nan,np.nan,np.nan])
                RS.append([np.nan,np.nan,np.nan])
                RE.append([np.nan,np.nan,np.nan])
                RW.append([np.nan,np.nan,np.nan])
                TR.append([np.nan,np.nan,np.nan]) 
                RI.append([np.nan,np.nan,np.nan])
                LI.append([np.nan,np.nan,np.nan])
                pass 
            
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

print(frames)
# quit()
for key,value in pose_land_marks.items():    
    for j in range(3):
        data=[]
        for i in range(frames):
            try:
                x=value[0][i][j]
                data.append(x)
            except:
                continue
        df[key+xyz[j]]=pd.Series(data)

print(df.info())
df.to_csv(pth+'\\mpipe_pre.csv',index=False)

try:
    # Read the JSON file containing the Session Directory
    with open('upperbody\gpane_dir.json', 'r') as file:
        session_data = json.load(file)

    # Get the directory path from the JSON data
    g_pth = session_data["gpane_directory"]

    # converting mpipe to mocap frame
    rotmat=[]
    org=[]
    with open(g_pth+'\\D435_rotmat.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            x=x.replace(']','')
            x=x.replace('[','')
            line=x.split(' ')
            while ' ' in line:
                line=line.remove(' ')
            while '' in line:
                ind=line.index('')
                line.pop(ind)
            x=[]
            for i in line:
                x.append(float(i))
            rotmat.append(x)
        rotmat=np.array(rotmat)

    with open(g_pth+'\\D435_org.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            x=x.replace(']','')
            x=x.replace('[','')
            org.append([float(x)])
        k_org=np.array(org)

    for index,j in df.iterrows():
        for k in range(1,1+7*3,3):
            point=[]
            for p in range(k,k+3):
                point.append(j[p])
            converted_point=frame_con(point,rotmat,org)
            # print(converted_point)
            for o in range(3):
                df.iloc[index,k+o]=converted_point[o]
except:
    pass

# Define columns to perform constant interpolation on
interpolate_columns = ['LS_x','LS_y','LS_z','RS_x','RS_y','RS_z','TR_x','TR_y','TR_z'] # df.columns.tolist() 

# Perform constant interpolation
df[interpolate_columns] = df[interpolate_columns].fillna(method='ffill')

# Iterate through all columns and applying cubic interpolation
for column in df.columns[1:]:
    column_series = df[column]
    column_series = column_series.interpolate(method='spline', order=3, s=0.,limit_direction='both')
    df[column] = column_series

# Saving the 3D points of each landmark
xyz=['x','y','z']
ls,le,lw,rs,re,rw=[],[],[],[],[],[]

c=1
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])   
        point=np.array(point)      
        ls.append(point)
c+=3
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])  
        point=np.array(point)      
        le.append(point)
c+=3
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])
        point=np.array(point)        
        lw.append(point)
c+=3
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])
        point=np.array(point)        
        rs.append(point)
c+=3
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])
        point=np.array(point)        
        re.append(point)
c+=3
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])
        point=np.array(point)        
        rw.append(point)

# Converting to numpy arrays
ls=np.array(ls)
le=np.array(le)
lw=np.array(lw)

rs=np.array(rs) 
re=np.array(re)
rw=np.array(rw)

# Finding the distances between the landmarks
lu=list(ls-le) # Left upper arm
ll=list(le-lw) # Left lower arm
ru=list(rs-re) # Rigth upper arm
rl=list(re-rw) # Right lower arm
ss=list(rs-ls) # Biacromial length / distance between shoulder

for i in range(len(lu)):
    lu[i]=np.linalg.norm(lu[i])
    ll[i]=np.linalg.norm(ll[i])
    ru[i]=np.linalg.norm(ru[i])
    rl[i]=np.linalg.norm(rl[i])
    ss[i]=np.linalg.norm(ss[i])

# Saving the values in a dataframe
df_ll=pd.DataFrame(columns=['lu','ll','ru','rl','ss'])

df_ll['lu']=pd.Series(lu)
df_ll['ll']=pd.Series(ll)
df_ll['ru']=pd.Series(ru)
df_ll['rl']=pd.Series(rl)
df_ll['ss']=pd.Series(ss)

# occlusion based on limb length
olu,oll,oru,orl,oss=[0],[0],[0],[0],[0]
th=0.10 # 10cm
for i in range(1,len(df_ll)):
    if abs(df_ll['lu'][i]-df_ll['lu'].mean())>th:
        olu.append(1)
    else:
        olu.append(0)
    if abs(df_ll['ll'][i]-df_ll['ll'].mean())>th:
        oll.append(1)
    else:
        oll.append(0)
    if abs(df_ll['ru'][i]-df_ll['ru'].mean())>th:
        oru.append(1)
    else:
        oru.append(0)
    if abs(df_ll['rl'][i]-df_ll['rl'].mean())>th:
        orl.append(1)
    else:
        orl.append(0)
    if abs(df_ll['ss'][i]-df_ll['ss'].mean())>th:
        oss.append(1)
    else:
        oss.append(0)

# filtering occlusion limb length
for index,j in df.iterrows():
    
    for k in range(3):
        if oss[index]==0 and olu[index]==0:
            lsx=df['LS_x'].iloc[index]
            lsy=df['LS_y'].iloc[index]
            lsz=df['LS_z'].iloc[index]
        else:
            lsx=np.nan
            lsy=np.nan
            lsz=np.nan
        df['LS_x'].iloc[index]=lsx
        df['LS_y'].iloc[index]=lsy
        df['LS_z'].iloc[index]=lsz
        
    for k in range(3):
        if oll[index]==0 and olu[index]==0:
            lex=df['LE_x'].iloc[index]
            ley=df['LE_y'].iloc[index]
            lez=df['LE_z'].iloc[index]
        else:
            lex=np.nan
            ley=np.nan
            lez=np.nan
        df['LE_x'].iloc[index]=lex
        df['LE_y'].iloc[index]=ley
        df['LE_z'].iloc[index]=lez

    for k in range(3):
        if oll[index]==0:
            lwx=df['LW_x'].iloc[index]
            lwy=df['LW_y'].iloc[index]
            lwz=df['LW_z'].iloc[index]
        else:
            lwx=np.nan
            lwy=np.nan
            lwz=np.nan
        df['LW_x'].iloc[index]=lwx
        df['LW_y'].iloc[index]=lwy
        df['LW_z'].iloc[index]=lwz

    for k in range(3):
        if oss[index]==0 and oru[index]==0:
            rsx=df['RS_x'].iloc[index]
            rsy=df['RS_y'].iloc[index]
            rsz=df['RS_z'].iloc[index]
        else:
            rsx=np.nan
            rsy=np.nan
            rsz=np.nan
        df['RS_x'].iloc[index]=rsx
        df['RS_y'].iloc[index]=rsy
        df['RS_z'].iloc[index]=rsz

    for k in range(3):
        if orl[index]==0 and oru[index]==0:
            rex=df['RE_x'].iloc[index]
            rey=df['RE_y'].iloc[index]
            rez=df['RE_z'].iloc[index]
        else:
            rex=np.nan
            rey=np.nan
            rez=np.nan
        df['RE_x'].iloc[index]=rex
        df['RE_y'].iloc[index]=rey
        df['RE_z'].iloc[index]=rez

    for k in range(3):
        if orl[index]==0 and oru[index]==0:
            rwx=df['RW_x'].iloc[index]
            rwy=df['RW_y'].iloc[index]
            rwz=df['RW_z'].iloc[index]
        else:
            rwx=np.nan
            rwy=np.nan
            rwz=np.nan
        df['RW_x'].iloc[index]=rwx
        df['RW_y'].iloc[index]=rwy
        df['RW_z'].iloc[index]=rwz
  
print(len(df))
# Perform constant interpolation
df[df.columns.tolist()] = df[df.columns.tolist()].fillna(method='ffill')
    
# # Iterate through all columns and applying cubic interpolation
# for column in df.columns[1:19]:
#     column_series = df[column]
#     column_series = column_series.interpolate(method='spline', order=3, s=0.,limit_direction='both')
    # df[column] = column_series

# applying savgol filter to data 
df_filtered = pd.DataFrame(savgol_filter(df, int(len(df)/100) * 2 + 3, 3, axis=0),
                                columns=df.columns,
                                index=df.index)

df_filtered['epoch_time'] = df['epoch_time'].values

print(df.info())

df.to_csv(pth+'\\mpipe.csv',index=False)
df_filtered.to_csv(pth+"\\mpipe_filtered.csv",index=False)

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Print the elapsed time
print(f"Program executed in {elapsed_time:.2f} seconds")
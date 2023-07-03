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
import re
import json

# Read the JSON file containing the Session Directory
with open('session_directory.json', 'r') as file:
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
    print(i)
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
                LS.append([dic[11]['x']*w,dic[11]['y']*h])
            except:
                LS.append(np.nan)
            try:
                LE.append([dic[13]['x']*w,dic[13]['y']*h])
            except:
                LE.append(np.nan)
            try:
                LW.append([dic[15]['x']*w,dic[15]['y']*h])
            except:
                LW.append(np.nan)
            try:
                RS.append([dic[12]['x']*w,dic[12]['y']*h])
            except:
                RS.append(np.nan)
            try:
                RE.append([dic[14]['x']*w,dic[14]['y']*h])
            except:
                RE.append(np.nan)
            try:
                RW.append([dic[16]['x']*w,dic[16]['y']*h])
            except:
                RW.append(np.nan)
            
            try:
                Smid=midpoint([dic[11]['x']*w,dic[11]['y']*h],[dic[12]['x']*w,dic[12]['y']*h])
                perpx=int(Smid[0])
                perpy=(int(Smid[1])+25)

                cv2.circle(color_image,(perpx,perpy) , 5, (0, 0, 255), 2)
                TR.append([perpx,perpy])     #in uv format  
            except:
                TR.append(np.nan)

            try:
                RI.append([dic[20]['x']*w,dic[20]['y']*h])
                LI.append([dic[19]['x']*w,dic[19]['y']*h])

                # Drawing the boxes around limbs for occlusion
                draw_box(color_image,[dic[11]['x']*w,dic[11]['y']*h],[dic[13]['x']*w,dic[13]['y']*h])
                draw_box(color_image,[dic[12]['x']*w,dic[12]['y']*h],[dic[14]['x']*w,dic[14]['y']*h])
                draw_box(color_image,[dic[13]['x']*w,dic[13]['y']*h],[dic[15]['x']*w,dic[15]['y']*h])
                draw_box(color_image,[dic[14]['x']*w,dic[14]['y']*h],[dic[16]['x']*w,dic[16]['y']*h])
                draw_box(color_image,[dic[16]['x']*w,dic[16]['y']*h],([dic[20]['x']*w,dic[20]['y']*h]),(255,0,255),40)
                draw_box(color_image,[dic[15]['x']*w,dic[15]['y']*h],([dic[19]['x']*w,dic[19]['y']*h]),(0,255,255),40) 
            except:
                RI.append(np.nan)
                LI.append(np.nan)
        except:
            LS.append(np.nan)
            LE.append(np.nan)
            LW.append(np.nan)

            RS.append(np.nan)
            RE.append(np.nan)
            RW.append(np.nan)

            TR.append(np.nan) 
            RI.append(np.nan)
            LI.append(np.nan)
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


pos = []
c=0
for i in campth:
    print(i)
    depth_file = open(i, "rb")
    unpacker = None
    unpacker = msgp.Unpacker(depth_file, object_hook=mpn.decode)
    for unpacked in unpacker:
        c+=1
        unpacked=np.asanyarray(unpacked) 
        pos.append(unpacked)
    depth_file.close()

cv2.destroyAllWindows()
pos=np.array(pos)
print(pos.shape)

# Dictionary containing landmark names and corresponding values
land_marks = {'LS': LS, 'LE': LE, 'LW': LW, 'RS': RS, 'RE': RE, 'RW': RW, 'TR': TR}

# pandas dataframe to hold landmark values
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

# take the stuff out of memory
pos = None
# Finding and correcting occlusions based on boxes
startflag = 0  # Flag to track the starting frame of occlusion
before_occ = 0  # Frame before occlusion
do = False  # Flag indicating whether occlusion is occurring or not

box_dic = {
    'lub': [LS, LE, ['LS', 'LE']],
    'rub': [RS, RE, ['RS', 'RE']],
    'llb': [LE, LW, ['LE', 'LW']],
    'rlb': [RE, RW, ['RE', 'RW']],
    'lhb': [LW, LI, ['LW']],
    'rhb': [RW, RI, ['RW']]
}

# Iterate through each landmark and associated box in the dictionary
for k, j in land_marks.items():
    for key, values in box_dic.items():
        # Iterate through each frame
        for i in range(c):
            r = 40 if key == 'lhb' or key == 'rhb' else 30  # Radius for drawing the box

            try:
                # Check if the landmark is inside the box and not already occluded
                if point_in_quad(j[i], draw_box(color_image, values[0][i], values[1][i], (0, 0, 1), r)) and k not in values[2]:
                    if startflag == 0:
                        startflag = i
                        before_occ = startflag - 1  # Frame before occlusion

                    for p in values[2]:
                        if df[k+'_z'].tolist()[before_occ] > df[p+'_z'].tolist()[before_occ]:
                            print(k, 'is occluded by', key, p, 'at frame', i)
                            # Uncomment the following lines to correct the occlusion
                            # df.loc[i, k+'_x'] = df.loc[before_occ, k+'_y']
                            # df.loc[i, k+'_y'] = df.loc[before_occ, k+'_x']
                            df.loc[i, k+'_z'] = df.loc[before_occ, k+'_z']

                    do = True  # Set occlusion flag to True
            except:
                pass
            else:
                do = False  # Set occlusion flag to False

        if do:
            startflag = 0  # Reset the start flag to 0 for the next occlusion check


# Read the JSON file containing the Session Directory
with open('gpane_dir.json', 'r') as file:
    session_data = json.load(file)

# Get the directory path from the JSON data
g_pth = session_data["gpane_directory"]

# converting mpipe to mocap frame
rotmat=[]
org=[]
with open(g_pth+'D435_rotmat.txt', 'r') as fp:
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

with open(g_pth+'D435_org.txt', 'r') as fp:
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


# Saving the 3D points of each landmark
xyz=['x','y','z']
ls,le,lw,rs,RE,rw=[],[],[],[],[],[]

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
        RE.append(point)
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
RE=np.array(RE)
rw=np.array(rw)

# Finding the distances between the landmarks
lu=list(ls-le) # Left upper arm
ll=list(le-lw) # Left lower arm
ru=list(rs-RE) # Rigth upper arm
rl=list(RE-rw) # Right lower arm
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
        df['LS_x'].iloc[index]=lsx
        df['LS_y'].iloc[index]=lsy
        df['LS_z'].iloc[index]=lsz
        
    for k in range(3):
        if oll[index]==0 and olu[index]==0:
            lex=df['LE_x'].iloc[index]
            ley=df['LE_y'].iloc[index]
            lez=df['LE_z'].iloc[index]
        df['LE_x'].iloc[index]=lex
        df['LE_y'].iloc[index]=ley
        df['LE_z'].iloc[index]=lez

    for k in range(3):
        if oll[index]==0:
            lwx=df['LW_x'].iloc[index]
            lwy=df['LW_y'].iloc[index]
            lwz=df['LW_z'].iloc[index]
        df['LW_x'].iloc[index]=lwx
        df['LW_y'].iloc[index]=lwy
        df['LW_z'].iloc[index]=lwz

    for k in range(3):
        if oss[index]==0 and oru[index]==0:
            rsx=df['RS_x'].iloc[index]
            rsy=df['RS_y'].iloc[index]
            rsz=df['RS_z'].iloc[index]
        df['RS_x'].iloc[index]=rsx
        df['RS_y'].iloc[index]=rsy
        df['RS_z'].iloc[index]=rsz

    for k in range(3):
        if orl[index]==0 and oru[index]==0:
            rex=df['RE_x'].iloc[index]
            rey=df['RE_y'].iloc[index]
            rez=df['RE_z'].iloc[index]
        df['RE_x'].iloc[index]=rex
        df['RE_y'].iloc[index]=rey
        df['RE_z'].iloc[index]=rez

    for k in range(3):
        if orl[index]==0 and oru[index]==0:
            rwx=df['RW_x'].iloc[index]
            rwy=df['RW_y'].iloc[index]
            rwz=df['RW_z'].iloc[index]
        df['RW_x'].iloc[index]=rwx
        df['RW_y'].iloc[index]=rwy
        df['RW_z'].iloc[index]=rwz

print(df.head())

df.to_csv(pth+'\\mpipe.csv',index=False)
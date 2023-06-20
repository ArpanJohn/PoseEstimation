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

# Filtering for box based occlusion
startflag=0
before_occ=0
do=False
box_dic={'lul': [LH,LK,['LH','LK']],'rul': [RH,RK,['RH','RK']],
         'lll': [LK,LA,['LK','LA']],'rll': [RK,RA,['RK','RA']],
         'lfb': [LA,LT,['LA','LT']],'rfb': [RA,RT,['RA','RT']]}

for k,j in land_marks.items():
    for key,values in box_dic.items():
        for i in range(c):
            r= 40 if key == 'lhb' or key == 'rhb' else 30
            try:
                if point_in_quad(j[i],draw_box(color_image,values[0][i],values[1][i],(0,0,1),r)) and k not in values[2]:
                    if startflag == 0:
                        startflag = i
                        before_occ=startflag-1 # before occlusion
                    for p in values[2]:
                        if df[k+'_z'].tolist()[before_occ]>df[p+'_z'].tolist()[before_occ]:
                            print(k,'is occluded by', key, p , 'at frame', i) 
                            # df.loc[i,k+'_x']=df.loc[before_occ,k+'_y']
                            # df.loc[i,k+'_y']=df.loc[before_occ,k+'_x']
                            df.loc[i,k+'_z']=df.loc[before_occ,k+'_z']
                    do=True
            except:
                pass
            else:
                do=False
        if do:
            startflag=0


# converting mpipe to mocap frame
rotmat=[]
org=[]
with open('D435_rotmat.txt', 'r') as fp:
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

with open('D435_org.txt', 'r') as fp:
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


# Filtering based on limb length occlusion
LH, LK, LA, LT, RH, RK, RA, RT = [],[],[],[],[],[],[],[]

c=1
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])   
        point=np.array(point)      
        LH.append(point)
c+=3
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])
        point=np.array(point)        
        LK.append(point)
c+=3
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])
        point=np.array(point)        
        LA.append(point)
c+=3
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])
        point=np.array(point)        
        LT.append(point)
c+=3
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])  
        point=np.array(point)      
        RH.append(point)
c+=3
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])
        point=np.array(point)        
        RK.append(point)
c+=3
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])
        point=np.array(point)        
        RA.append(point)
c+=3
for i,j in df.iterrows():
    for k in range(c,c+3,3):
        point=[]
        for p in range(k,k+3):
            point.append(j[p])
        point=np.array(point)        
        RT.append(point)

LH=np.array(LH)
LK=np.array(LK)
LA=np.array(LA)
LT=np.array(LT)

RH=np.array(RH)
RK=np.array(RK)
RA=np.array(RA)
RT=np.array(RT)

Lul=list(LH-LK)
Lll=list(LK-LA)
Lf=list(LA-LT)
Rul=list(RH-RK)
Rll=list(RK-RA)
Rf=list(RA-RT)
H=list(RH-LH)

for i in range(len(Lul)):
    Lul[i]=np.linalg.norm(Lul[i])
    Lll[i]=np.linalg.norm(Lll[i])
    Lf[i]=np.linalg.norm(Lf[i])
    Rul[i]=np.linalg.norm(Rul[i])
    Rll[i]=np.linalg.norm(Rll[i])
    Rf[i]=np.linalg.norm(Rf[i])
    H[i]=np.linalg.norm(H[i])

df_limb=pd.DataFrame(columns=['Lul','Lll','Lf','Rul','Rll','Rf','H'])

df_limb['Lul']=pd.Series(Lul)
df_limb['Lll']=pd.Series(Lll)
df_limb['Lf']=pd.Series(Lf)
df_limb['Rul']=pd.Series(Rul)
df_limb['Rll']=pd.Series(Rll)
df_limb['Rf']=pd.Series(Rf)
df_limb['H']=pd.Series(H)

oLul,oLll,oLf,oRul,oRll,oRf,oH=[0],[0],[0],[0],[0],[0],[0]
th=0.10 # 10cm
for i in range(1,len(df_limb)):
    if abs(df_limb['Lul'][i]-df_limb['Lul'].mean())>th:
        oLul.append(1)
    else:
        oLul.append(0)
    if abs(df_limb['Lll'][i]-df_limb['Lll'].mean())>th:
        oLll.append(1)
    else:
        oLll.append(0)
    if abs(df_limb['Lf'][i]-df_limb['Lf'].mean())>th:
        oLf.append(1)
    else:
        oLf.append(0)
    if abs(df_limb['Rul'][i]-df_limb['Rul'].mean())>th:
        oRul.append(1)
    else:
        oRul.append(0)
    if abs(df_limb['Rll'][i]-df_limb['Rll'].mean())>th:
        oRll.append(1)
    else:
        oRll.append(0)
    if abs(df_limb['Rf'][i]-df_limb['Rf'].mean())>th:
        oRf.append(1)
    else:
        oRf.append(0)
    if abs(df_limb['H'][i]-df_limb['H'].mean())>th:
        oH.append(1)
    else:
        oH.append(0)
    
for index,j in df.iterrows():
    
    for k in range(3):
        if oLul[index]==0 and oH[index]==0:
            LHx=df['LH_x'].iloc[index]
            LHy=df['LH_y'].iloc[index]
            LHz=df['LH_z'].iloc[index]
        df['LH_x'].iloc[index]=LHx
        df['LH_y'].iloc[index]=LHy
        df['LH_z'].iloc[index]=LHz
        
    for k in range(3):
        if oLll[index]==0 and oLul[index]==0:
            LKx=df['LK_x'].iloc[index]
            LKy=df['LK_y'].iloc[index]
            LKz=df['LK_z'].iloc[index]
        df['LK_x'].iloc[index]=LKx
        df['LK_y'].iloc[index]=LKy
        df['LK_z'].iloc[index]=LKz

    for k in range(3):
        if oLll[index]==0 and oLf[index]==0:
            LAx=df['LA_x'].iloc[index]
            LAy=df['LA_y'].iloc[index]
            LAz=df['LA_z'].iloc[index]
        df['LA_x'].iloc[index]=LAx
        df['LA_y'].iloc[index]=LAy
        df['LA_z'].iloc[index]=LAz

    for k in range(3):
        if oLf[index]==0:
            LFx=df['LT_x'].iloc[index]
            LFy=df['LT_y'].iloc[index]
            LFz=df['LT_z'].iloc[index]
        df['LT_x'].iloc[index]=LFx
        df['LT_y'].iloc[index]=LFy
        df['LT_z'].iloc[index]=LFz

    for k in range(3):
            if oRul[index]==0 and oH[index]==0:
                RHx=df['RH_x'].iloc[index]
                RHy=df['RH_y'].iloc[index]
                RHz=df['RH_z'].iloc[index]
            df['RH_x'].iloc[index]=RHx
            df['RH_y'].iloc[index]=RHy
            df['RH_z'].iloc[index]=RHz
        
    for k in range(3):
        if oRll[index]==0 and oRul[index]==0:
            RKx=df['RK_x'].iloc[index]
            RKy=df['RK_y'].iloc[index]
            RKz=df['RK_z'].iloc[index]
        df['RK_x'].iloc[index]=RKx
        df['RK_y'].iloc[index]=RKy
        df['RK_z'].iloc[index]=RKz


    for k in range(3):
        if oRll[index]==0 and oRf[index]==0:
            RAx=df['RA_x'].iloc[index]
            RAy=df['RA_y'].iloc[index]
            RAz=df['RA_z'].iloc[index]
        df['RA_x'].iloc[index]=RAx
        df['RA_y'].iloc[index]=RAy
        df['RA_z'].iloc[index]=RAz


    for k in range(3):
        if oRf[index]==0:
            RFx=df['RT_x'].iloc[index]
            RFy=df['RT_y'].iloc[index]
            RFz=df['RT_z'].iloc[index]
        df['RT_x'].iloc[index]=RFx
        df['RT_y'].iloc[index]=RFy
        df['RT_z'].iloc[index]=RFz


df.to_csv('mpipe.csv')
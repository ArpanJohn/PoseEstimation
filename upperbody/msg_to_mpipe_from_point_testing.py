#getting the data and running the model
import numpy as np
import cv2 #type:ignore
import msgpack as msgp
import msgpack_numpy as mpn#type:ignore
import glob
import os
import time
import mediapipe as mp #type:ignore
from support.funcs import *
import pandas as pd #type:ignore
from natsort import natsorted #type:ignore
import re as re_str
import json
import time
import matplotlib.pyplot as plt #type:ignore
from scipy.signal import savgol_filter #type:ignore
from scipy.interpolate import CubicSpline, interp1d #type:ignore

def make_list_ascending(lst):
    indexes = []
    for i in range(len(list(lst)) - 1):
        if lst[i] >= lst[i + 1]:
            indexes.append(i + 1)
            lst[i + 1] += 0.001
    return lst

# Measure the execution time
start_time = time.time()

# Read the JSON file containing the Session Directory
with open('upperbody\SessionDirectory.json', 'r') as file:
    session_data = json.load(file)

# Get the directory path from the JSON data
pth = session_data["directory"]

# Getting the data
df=pd.read_csv(pth+'\\mpipe_pre.csv')
# drop repeating rows in mpipe
df = df.drop_duplicates(subset=df.columns[0])
df3=df.copy()
df2=df.copy()

print(df3.info())

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
            for o in range(3):
                df.iloc[index,k+o]=converted_point[o]
except:
    pass

df.dropna(inplace=True) 
# Extract the x values from df_mpipe and df2
x = df['epoch_time']
x=make_list_ascending(list(x))
x_new=df3['epoch_time']

# Loop through the columns of df2 (excluding epoch_time)
for column in df.columns[1:]:
    if column in ['LS_x','LS_y','LS_z','RS_x','RS_y','RS_z','TR_x','TR_y','TR_z']:
        k ='zero'
    else:
        k='cubic'

    # Extract the y values from df_mpipe for the current column
    y = df[column]
    
    # Generate the interpolating function
    interpolating = interp1d(x,y,kind = k,fill_value='extrapolate')
    
    # Interpolate y values onto df2 for the current column
    interpolated_values = interpolating(x_new)
    
    # Add the interpolated values as a new column in df2
    df2['interpolated_' + column] = interpolated_values
    
df2.drop(df.columns,axis=1,inplace=True)

# Loop through the columns of the DataFrame
for column in df2.columns:
    # Remove the desired string from column names using str.replace()
    new_column_name = column.replace('interpolated_', '')

    # Rename the column with the updated name
    df2.rename(columns={column: new_column_name}, inplace=True)

df2.insert(0, 'epoch_time', df['epoch_time'])

df=df2.copy()
df.reset_index(inplace = True)
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
th=10 # 10cm

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


df2=df.copy()
df.dropna(inplace=True) 
# Extract the x values from df_mpipe and df2
x = df['epoch_time']
x=make_list_ascending(list(x))
x_new=df3['epoch_time']

# Loop through the columns of df2 (excluding epoch_time)
for column in df.columns[1:]:
    if column in ['LS_x','LS_y','LS_z','RS_x','RS_y','RS_z','TR_x','TR_y','TR_z']:
        k ='zero'
    else:
        k='cubic'

    # Extract the y values from df_mpipe for the current column
    y = df[column]
    
    # Generate the interpolating function
    interpolating = interp1d(x,y,kind = k,fill_value='extrapolate')
    
    # Interpolate y values onto df2 for the current column
    interpolated_values = interpolating(x_new)
    
    # Add the interpolated values as a new column in df2
    df2['interpolated_' + column] = interpolated_values
    
df2.drop(df.columns,axis=1,inplace=True)

# Loop through the columns of the DataFrame
for column in df2.columns:
    # Remove the desired string from column names using str.replace()
    new_column_name = column.replace('interpolated_', '')

    # Rename the column with the updated name
    df2.rename(columns={column: new_column_name}, inplace=True)

try:
    df2.insert(0, 'epoch_time', df['epoch_time'])
except :
    pass
df=df2.copy()

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
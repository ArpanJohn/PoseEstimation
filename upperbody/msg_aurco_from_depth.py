import numpy as np
import cv2
import cv2.aruco as aruco
import msgpack as msgp
import msgpack_numpy as mpn
import glob
import os
import matplotlib.pyplot as plt
from natsort import natsorted
import re

pth = r"C:\Users\arpan\OneDrive\Documents\internship\rec_program\savdir\d"

lst = os.listdir(pth)
vid_name = lst[-1]

targetPattern = f"{pth}\\DEPTH*"
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

height = int(h)
width = int(w) 

# CX_DEPTH = float(params[6])
# CY_DEPTH = float(params[7])
# FX_DEPTH = float(params[3])
# FY_DEPTH = float(params[4])

CX_DEPTH = 326.143 
CY_DEPTH = 234.082
FX_DEPTH = 383.223
FY_DEPTH = 383.223

for unpacked in unpacker:
    timestamps.append(unpacked)

rec_dur=timestamps[-1]-timestamps[0]

# Print the parameters of the recording
# print(('recording duration '+f"{rec_dur:.3}"+' s'+'\nresolution :'+str(w)+'x'+str(h)+ '; fps : '+str(fps)))
# print('number of frames:', len(timestamps))

# Setting the image as the middle frame of video
aurco_flag=int(len(timestamps)/2)

# Sorting the color and depth files
cpth=natsorted(cpth)
campth=natsorted(campth)

# compute indices:
jj = np.tile(range(width), height)
ii = np.repeat(range(height), width)

# Compute constants:
xx = (jj - CX_DEPTH) / FX_DEPTH
yy = (ii - CY_DEPTH) / FY_DEPTH
# transform depth image to vector of z:
length = height * width

# Initializing frame counter
frames=0


for i,j in zip(cpth,campth):
    # print(j)
    col_file = open(i, "rb")
    unpacker = None
    unpacker = msgp.Unpacker(col_file, object_hook=mpn.decode)
    depth_file = open(j, "rb")
    d_unpacker = None
    d_unpacker = msgp.Unpacker(depth_file, object_hook=mpn.decode)
    for unpacked,d_unpacked in zip(unpacker,d_unpacker):
        frames+=1

        if frames == aurco_flag:  
            depth_image=np.asanyarray(d_unpacked)

            # compute indices:
            jj = np.tile(range(width), height)
            ii = np.repeat(range(height), width)
            # Compute constants:
            xx = (jj - CX_DEPTH) / FX_DEPTH
            yy = (ii - CY_DEPTH) / FY_DEPTH
            # transform depth image to vector of z:
            length = height * width
            d = (np.asanyarray(depth_image).reshape(height * width)) / 1000
            z = d
            # compute point cloud
            pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))
            pcd=pcd.reshape(height,width,3)

            # Load the image
            image = (np.asanyarray(unpacked))

           # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Specify the ArUco dictionary
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

            # Create the parameters for the  ArUco detector
            parameters = aruco.DetectorParameters_create()

            # Detect the ArUco markers in the image
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            for corner in corners:
                    center = np.mean(corner[0], axis=0)
                    x = int(center[0])
                    y = int(center[1])
                    cv2.circle(image, (x, y), 50, (0, 255, 0), 2)

            # Draw a circle around the center of each detected marker
            centers=[]
            for corner in corners:
                    center = np.mean(corner[0], axis=0)
                    x = int(center[0])
                    y = int(center[1])
                    centers.append((x,y))
                    cv2.circle(image, (x, y), 50, (0, 255, 0), 2)

            # Show the result
            # cv2.imshow("Result", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    depth_file.close()
    col_file.close()


cv2.destroyAllWindows()

# cento=pcd[centers[2][1]][centers[2][0]][1]
# centz=pcd[centers[0][1]][centers[0][0]][1]
# centx=pcd[centers[1][1]][centers[1][0]][1]

# print(cento)
# print(centx)
# print(centz)
# quit()

cento=pcd[centers[2][1]][centers[2][0]]
centz=pcd[centers[0][1]][centers[0][0]]
centx=pcd[centers[1][1]][centers[1][0]]

print(np.linalg.norm(cento-centz)*100)
print(np.linalg.norm(cento-centx)*100)
print(np.linalg.norm(centz-centx)*100)


# Assigning centers to origin, x axis and z axis
for i in range(3):
    for j in range(3):
        if i!=j:
            if 18.5<np.linalg.norm(pcd[centers[i][1]][centers[i][0]]-pcd[centers[j][1]][centers[j][0]])*100<21.5 and 13.5<np.linalg.norm(pcd[centers[3-i-j][1]][centers[3-j-i][0]]-pcd[centers[j][1]][centers[j][0]])*100<16.5:
                cento=pcd[centers[j][1]][centers[j][0]]
                centz=pcd[centers[i][1]][centers[i][0]]
                centx=pcd[centers[3-j-i][1]][centers[3-j-i][0]]
                print('Centers assigned')

quit()

#verifiying centers
org_z=np.add(centz,-cento)*100
org_x=np.add(cento,-centx)*100
print(np.linalg.norm(org_z))
print(np.linalg.norm(org_x))

# Finding the Rotation matrix
v1 = centx - cento  # v1
v2 = centz - cento  # v2

vxnorm = v1 / np.linalg.norm(v1)
vzcap = v2 - (vxnorm.T @ v2) * vxnorm
vznorm = vzcap / np.linalg.norm(vzcap)

vynorm=np.cross(vznorm.T,vxnorm.T).reshape(3,1)

vznorm=vznorm.reshape(3,1)
vxnorm=vxnorm.reshape(3,1)

rotMat = np.hstack((vxnorm, vynorm, vznorm))

print('rotmat : ',rotMat)
print('origin : ',cento)

# Saving the roation Matrix and the origin to files
with open(r'upperbody\D435_rotmat.txt', 'w') as fp:
    for item in rotMat:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'upperbody\D435_org.txt', 'w') as fp:
    for item in cento:
        # write each item on a new line
        fp.write("%s\n" % item)



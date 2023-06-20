import numpy as np
import cv2
import cv2.aruco as aruco
import msgpack as msgp
import msgpack_numpy as mpn
import glob
import os
import matplotlib.pyplot as plt

pth = 'path'

lst = os.listdir(pth)
vid_name = lst[-1]

# targetPattern = f"{pth}\\CAMSPACE*" 
targetPattern = f"{pth}\\DEPTH*"
campth = glob.glob(targetPattern)

targetPattern_param = f"{pth}\\PARAM*"
ppth = glob.glob(targetPattern_param)

targetPattern_colour = f"{pth}\\COLOUR*"
cpth = glob.glob(targetPattern_colour)

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
         
         img.append(unpacked)
        #  cv2.imshow("aurco", unpacked)
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

print(len(img))

pos = []
c=0
for i in campth:
    print(i)
    depth_file = open(i, "rb")
    unpacker = None
    unpacker = msgp.Unpacker(depth_file, object_hook=mpn.decode)
    for unpacked in unpacker:
         c+=1       
         unpacked=cv2.flip(unpacked,1)

         pos.append(unpacked)
         if np.all(pos[-1]) == -1:
             cv2.destroyAllWindows()
             break
        #  cv2.imshow("aurco", unpacked)
         if cv2.waitKey(1) & 0xFF == ord('q'):
             break
         try:
            if (unpacked)==-1:
                cv2.destroyAllWindows()
                break
         except:
             continue
    depth_file.close()

cv2.destroyAllWindows()

aurco_flag=int(len(img)/2)

# Load the image
image = (img[aurco_flag])

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Specify the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

# Create the parameters for the ArUco detector
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
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cento=pos[aurco_flag][centers[2][1],centers[2][0]]
centz=pos[aurco_flag][centers[0][1],centers[0][0]]
centx=pos[aurco_flag][centers[1][1],centers[1][0]]

# Assigning centers to origin, x axis and z axis
for i in range(3):
    for j in range(3):
        if i!=j:
            if 18.5<np.linalg.norm(pos[aurco_flag][centers[i][1],centers[i][0]]-pos[aurco_flag][centers[j][1],centers[j][0]])*100<21.5 and 13.5<np.linalg.norm(pos[aurco_flag][centers[3-i-j][1],centers[3-j-i][0]]-pos[aurco_flag][centers[j][1],centers[j][0]])*100<16.5:
                cento=pos[aurco_flag][centers[j][1],centers[j][0]]
                centz=pos[aurco_flag][centers[i][1],centers[i][0]]
                centx=pos[aurco_flag][centers[3-j-i][1],centers[3-j-i][0]]
                print('Centers assigned')

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
with open(r'D435_rotmat.txt', 'w') as fp:
    for item in rotMat:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'D435_org.txt', 'w') as fp:
    for item in cento:
        # write each item on a new line
        fp.write("%s\n" % item)



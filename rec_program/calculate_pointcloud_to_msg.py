# Importing necessary libraries
import numpy as np
import cv2
import msgpack as msgp
import msgpack_numpy as mpn
import os
import os.path
import time
from support.funcs import *
from datetime import date, datetime
from numpy import random
import threading
from queue import Queue
import glob
import matplotlib.pyplot as plt
import re

def createFile(SessDir, fileCounter):
    # Getting the file names
    POINTname = SessDir + "/" + "POINT" + "_" + str(fileCounter) + ".msgpack"
    
    # Opening depth and colour file for writing
    POINTfile = open(POINTname, 'wb')

    print(f"creating files {fileCounter}")

    return POINTfile

def save_frames(pointcloud, POINTfile):
    # saving the depth information
    pc_packed = msgp.packb(pointcloud, default=mpn.encode)
    POINTfile.write(pc_packed)

# Path to session folder
pth = r"C:\Users\arpan\OneDrive\Documents\internship\rec_program\savdir\Session_26-06-23_15-12-16_2451"

# Getting the COLOR files
targetPattern_colour = f"{pth}\\COLOUR*"
cpth = glob.glob(targetPattern_colour)

# Gettin the DEPTH files
targetPattern = f"{pth}\\DEPTH*"
campth = glob.glob(targetPattern)

# Getting the parameter file
targetPattern_param = f"{pth}\\PARAM*"
ppth = glob.glob(targetPattern_param)

#obtaining parameters and list time_stamps
p = open(ppth[0], "rb")
unpacker=None
unpacker = list(msgp.Unpacker(p, object_hook=mpn.decode))
timestamps = []
f=0
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

w = params[0]
h = params[1]
fps = params[-1]

CX_DEPTH = float(params[3])
CY_DEPTH = float(params[4])
FX_DEPTH = float(params[6])
FY_DEPTH = float(params[7])

for unpacked in unpacker:
    timestamps.append(unpacked)

rec_dur=timestamps[-1]-timestamps[0]

# Print the parameters of the recording
print(('recording duration '+f"{rec_dur:.3}"+' s'+'\nresolution :'+str(w)+'x'+str(h)+ '; fps : '+str(fps)))

c=0

height = int(h)
width = int(w) 

# compute indices:
jj = np.tile(range(width), height)
ii = np.repeat(range(height), width)
# Compute constants:
xx = (jj - CX_DEPTH) / FX_DEPTH
yy = (ii - CY_DEPTH) / FY_DEPTH
# transform depth image to vector of z:
length = height * width

# Setting counters
counter = 1
fileCounter = 1

# Creating necessary files
POINTfile=createFile(pth,fileCounter)

for i in campth:
    depth_file = open(i, "rb")
    unpacker = None
    unpacker = msgp.Unpacker(depth_file, object_hook=mpn.decode)
    for unpacked in unpacker:
        z = np.asanyarray(unpacked).reshape(height * width)
        # compute point cloud
        pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))
        c+=1

        # Saving frames to msgpack files
        save_frames(pcd,POINTfile)

        # Counting frames in each msgpack
        counter = counter + 1

        # When 90 frames in one .msgpack file, open a new file
        if counter == 90:
            fileCounter = fileCounter + 1
            POINTfile.close()
            POINTfile=createFile(pth,fileCounter)
            counter = 1   
    depth_file.close()







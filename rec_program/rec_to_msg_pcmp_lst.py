# Importing necessary libraries
import pyrealsense2 as rs
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
import json

import keyboard

# getting Date and time
now = datetime.now()
today = date.today()

# initializing things
SessDir=[]

class recorder():
    def __init__(self):
        # Setting the parameters of the stream
        self.h = 720  
        self.w = 1280 
        # self.h = 480 
        # self.w = 640 
        self.fps=30
        self.f=0

        # Initializing lists of color, depth, and parameters
        self.color_image_list=[]
        self.depth_frame_list=[]
        self.stream_parm_list=[]

        # Stop flag
        self.stop_flag=False
        # initializing base parameters
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        self.pc = rs.pointcloud()


    def run(self):
        t1 = threading.Thread(target=self.readframe)
        t2 = threading.Thread(target=self.processing_and_saving)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        print('threads done')

    def processing_and_saving(self):
        # running the the second thread for calculating pointcloud and saving            
        # Setting counters
        self.counter = 1
        self.fileCounter = 1

        # Creating necessary files
        self.createFile(self.fileCounter)

        # Saving stream parameters to parameters file
        parameters=self.stream_parm_list.pop(0)
        p_packed = msgp.packb(str(parameters)+str(self.fps))
        self.paramFile.write(p_packed)


        while not self.stop_flag:
            if not len(self.color_image_list) == 0 and not len(self.depth_frame_list) == 0 and not len(self.stream_parm_list) == 0:
                # print number of items in queue
                # print('Queue size : ', color_image_queue.qsize(), end = '\r')

                # Getting the frames from lists
                color_image=self.color_image_list.pop(0)
                depth_frame=self.depth_frame_list.pop(0)
                d_timestamp=self.stream_parm_list.pop(0)

                # Calculating the pointcloud
                try:
                    points = self.pc.calculate(depth_frame)
                    v = points.get_vertices()
                    verts = np.asanyarray(v).view(np.float32)
                    xyzpos=verts.reshape(self.h,self.w, 3)  # xyz
                    point_image=xyzpos.astype(np.float16)            
                except:
                    print('error in pointcloud calculation')


                # Saving frames to msgpack files
                self.save_frames(color_image, point_image, d_timestamp, self.colourfile, self.depthfile, self.paramFile)
                
                # Counting frames in each msgpack
                self.counter = self.counter + 1

                # When 90 frames in one .msgpack file, open a new file
                if self.counter == 90:
                    self.fileCounter = self.fileCounter + 1
                    self.colourfile.close()
                    self.depthfile.close()
                    self.createFile(self.fileCounter)
                    self.counter = 1   


        # saving the directory for saving the time graph
        SessDir.append(self.sessionDir)
        self.paramFile.close()

    def createFile(self, fileCounter):

        # gettin the time,date,etc for session directory and file names
        self.pFileName = 'test'
        self.tm1 = today.strftime("%d-%m-%y")
        self.tm2 = now.strftime("%H-%M-%S")
        self.tM = self.tm1 + " " + self.tm2
        self.tm = self.tm1 + "_" + self.tm2
        # Open the JSON file
        with open('rec_program\savdir_path.json') as json_file:
            data = json.load(json_file)
            # Access the directory path
            pth = data['directory_path']
        self.savingDir = pth 
        self.temp_dir = pth
        self.f=self.f+1

        # creating the files
        if fileCounter == 1:
            self.rnd = random.randint(999)
            self.sessionName = "Session_lst_" + self.tm1 + "_" + self.tm2 + "_" + str(self.rnd)+str(self.f)
            self.sessionDir = os.path.join(self.savingDir, self.sessionName)
            self.temp_save = os.path.join(self.temp_dir, self.sessionName)
            os.mkdir(self.sessionDir)

            self.parmsFileName = self.sessionDir + "/" + "PARAMS" + "_" + self.tm + "_" + str(
                self.rnd) + ".msgpack"

            self.paramFile = open(self.parmsFileName, 'wb')

        # Getting the file names
        self.commonName = self.pFileName + " " + self.tM + " " + str(self.rnd)
        self.depthfilename = self.sessionDir + "/" + "POINT" + "_" + self.tm + "_" + str(
            self.rnd) + "_" + str(fileCounter) + ".msgpack"
        self.colourfilename = self.sessionDir + "/" + "COLOUR" + "_" + self.tm + "_" + str(
            self.rnd) + "_" + str(fileCounter) + ".msgpack"
        
        # Opening depth and colour file for writing
        self.depthfile = open(self.depthfilename, 'wb')
        self.colourfile = open(self.colourfilename, 'wb')

        print(f"creating files {fileCounter}")

            
    def save_frames(self,colorImg, depthImg, milliseconds, colorFile, depthFile, paramsfile):
        # saving the depth information
        d_packed = msgp.packb(depthImg, default=mpn.encode)
        depthFile.write(d_packed)

        # saving the color information
        c_packed = msgp.packb(colorImg, default=mpn.encode)
        colorFile.write(c_packed)

        # saving the time information
        prm = milliseconds/1000 # converting miliseconds to seconds
        p_packed = msgp.packb(prm)
        paramsfile.write(p_packed)

    def readframe(self):
        # Starting the stream for depth/color cameras
        self.config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, self.fps)

        try:
            # Start streaming from file
            profile=self.pipeline.start(self.config)   

            # Getting depth intrinsics
            profiled = profile.get_stream(rs.stream.depth)
            profile1d=profiled.as_video_stream_profile() 
            intr = profile1d.get_intrinsics()   
            intr=str(intr) 
            self.stream_parm_list.append('depth intrinsics:'+intr) 

            # initializing alignment
            align_to = rs.stream.color
            align = rs.align(align_to)

            # Number of frames 
            c=0

            start_time = time.time()
            while time.time() - start_time < 200:
                # Checking if there are more frames
                frame_present, frameset = self.pipeline.try_wait_for_frames()
                    
                #End loop once video finishes
                if not frame_present:
                    continue

                # Wait for a coherent pair of frames: depth and color
                # frames = self.pipeline.wait_for_frames()

                # Aligning the depth and color frames
                aligned_frames = align.process(frameset)
                depth_frame = frameset.get_depth_frame()
                color_frame = frameset.get_color_frame()
                if not depth_frame or not color_frame:
                    continue                                

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame() 
                color_frame = aligned_frames.get_color_frame()

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data()) 

                # limit size of queues to 10
                while len(self.color_image_list)>10:
                    self.color_image_list.pop(0)
                    self.depth_frame_list.pop(0)
                    self.stream_parm_list.pop(0)      
                    print('binned')

                # putting the items in the lists for processing and saving
                self.color_image_list.append(color_image)
                self.depth_frame_list.append(aligned_depth_frame)
                self.stream_parm_list.append(rs.frame.get_frame_metadata(depth_frame,rs.frame_metadata_value.time_of_arrival))
                    
                # Show images
                cv2.imshow('RealSense', color_image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    self.stop_flag = True
                    break
                c+=1
                
        finally:
            # Stop streaming
            self.pipeline.stop()
            cv2.destroyAllWindows()
    
if __name__ == '__main__':

    thread = recorder()
    thread.run()  

# # Path to session folder
pth = SessDir[0]

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

CX_DEPTH = params[3]
CY_DEPTH = params[4]
FX_DEPTH = params[6]
FY_DEPTH = params[7]

for unpacked in unpacker:
    timestamps.append(unpacked)

rec_dur=timestamps[-1]-timestamps[0]

# Print the parameters of the recording
print(('recording duration '+f"{rec_dur:.3}"+' s'+'\nresolution :'+str(w)+'x'+str(h)+ '; fps : '+str(fps)))
print('number of frames:', len(timestamps))

correlation = np.corrcoef(range(len(timestamps)),timestamps)[0,1]

print(f'Pearson product moment correlation coefficient: {correlation}')

# Show the graph of time of arrival of each frame (should be linear)
plt.plot(range(len(timestamps)),timestamps)
plt.title(('recording duration '+f"{rec_dur:.3}"+' s'+'\n resolution :'+str(w)+'x'+str(h)+ '; fps : '+str(fps)+f'\nlinearity: {correlation:.8}'))
plt.xlabel('frame')
plt.ylabel('epoch time in seconds')
plt.savefig(pth+'/time_graph.jpg')
# plt.show()


# timer
# timer=int(input('enter timer'))

# while timer:  
#     array = np.zeros((720, 1280, 3),dtype = np.uint8)
#     blank_display=cv2.flip(array,1)
#     m, s = divmod(timer, 60)  
#     timeleft = '{:02d}:{:02d}'.format(m, s) 
#     print(timeleft, end="\r")  
#     time.sleep(1)  
#     timer = timer-1 

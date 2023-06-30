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

# getting Date and time
now = datetime.now()
today = date.today()

# initializing things
SessDir=[]
stop = Queue()

class rec(threading.Thread):
    def __init__(self):
        stop.put(False)

        # Setting the parameters of the stream
        self.h=480
        self.w=640
        self.fps=30
        self.f=0

        # initializing base parameters
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

    def run(self):
        # running the first thread for recording 
        print ("Starting " + self.name)
        if self.threadID == 1:
            self.readframe()

        # running the the second thread for calculating pointcloud and saving
        if self.threadID == 2:
            
            # Setting counters
            self.counter = 1
            self.fileCounter = 1

            # Creating necessary files
            self.createFile(self.fileCounter)

            # Saving stream parameters to parameters file
            parameters=param_queue.get()
            p_packed = msgp.packb(str(parameters)+str(self.fps))
            self.paramFile.write(p_packed)
        
            while not stop.get():
                # Get color image and depth frame from queue to calculate pointcloud and save to files     
                while not color_image_queue.empty():
                    # print number of items in queue
                    print('Queue size : ', color_image_queue.qsize(), end = '\r')
                    # Getting the frames from queue
                    color_image=color_image_queue.get()
                    depth_image=depth_image_queue.get()

                    # getting the time stamp of the depth frame
                    timestamp=param_queue.get()

                    # Saving frames to msgpack files
                    rec.save_frames(color_image, depth_image, timestamp, 
                                    self.colourfile, self.depthfile, self.paramFile)
                    
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
        print ("Exiting " + self.name)

    def createFile(self, fileCounter):

        # gettin the time,date,etc for session directory and file names
        self.pFileName = 'test'
        self.tm1 = today.strftime("%d-%m-%y")
        self.tm2 = now.strftime("%H-%M-%S")
        self.tM = self.tm1 + " " + self.tm2
        self.tm = self.tm1 + "_" + self.tm2
        self.savingDir = r'C:\Users\arpan\OneDrive\Documents\internship\rec_program\savdir'
        self.temp_dir = r'C:\Users\arpan\OneDrive\Documents\internship\rec_program\temp_dir'
        self.f=self.f+1

        # creating the files
        if fileCounter == 1:
            self.rnd = random.randint(999)
            self.sessionName = "Session_" + self.tm1 + "_" + self.tm2 + "_" + str(self.rnd)+str(self.f)
            self.sessionDir = os.path.join(self.savingDir, self.sessionName)
            self.temp_save = os.path.join(self.temp_dir, self.sessionName)
            os.mkdir(self.sessionDir)

            self.parmsFileName = self.sessionDir + "/" + "PARAMS" + "_" + self.tm + "_" + str(
                self.rnd) + ".msgpack"

            self.paramFile = open(self.parmsFileName, 'wb')

        # Getting the file names
        self.commonName = self.pFileName + " " + self.tM + " " + str(self.rnd)
        self.depthfilename = self.sessionDir + "/" + "DEPTH" + "_" + self.tm + "_" + str(
            self.rnd) + "_" + str(fileCounter) + ".msgpack"
        self.colourfilename = self.sessionDir + "/" + "COLOUR" + "_" + self.tm + "_" + str(
            self.rnd) + "_" + str(fileCounter) + ".msgpack"
        
        # Opening depth and colour file for writing
        self.depthfile = open(self.depthfilename, 'wb')
        self.colourfile = open(self.colourfilename, 'wb')

        print(f"creating files {fileCounter}")

            
    def save_frames(colorImg, depthImg, milliseconds, colorFile, depthFile, paramsfile):
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
            param_queue.put('depth intrinsics:'+intr) 

            # initializing alignment
            align_to = rs.stream.color
            align = rs.align(align_to)

            # Number of frames 
            c=0
            
            while True:
                # Checking if there are more frames
                frame_present, frameset = self.pipeline.try_wait_for_frames()
                    
                #End loop once video finishes
                if not frame_present:
                    break

                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()

                # Aligning the depth and color frames
                aligned_frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue                                

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame() 
                color_frame = aligned_frames.get_color_frame()

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data()) 
                depth_image = np.asanyarray(aligned_depth_frame.get_data()) 

                # Signal that thread1 is running
                stop.put(False)

                # limit size of queues to 10
                while color_image_queue.qsize()>10:
                    bin=color_image_queue.get()
                    bin=depth_image_queue.get()
                    bin=param_queue.get()
                    bin=None

                # putting the images in the queues
                color_image_queue.put(color_image)
                depth_image_queue.put(depth_image)

                # Putting timestamps in queue
                param_queue.put(rs.frame.get_frame_metadata(depth_frame,rs.frame_metadata_value.time_of_arrival))

                # resizing for display
                color_image=cv2.resize(color_image, (int(self.w*self.windowscale),int(self.h*self.windowscale)))
                    
                # Show images
                cv2.imshow('RealSense', color_image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    stop.put(True)
                    break
                c+=1
                
        finally:

            # Stop streaming
            self.pipeline.stop()
            cv2.destroyAllWindows()

threadLock = threading.Lock()
threads=[]

# Initializing the queues
color_image_queue, depth_image_queue=Queue(),Queue()
param_queue=Queue()
        
# Create new threads
thread1 = rec(1, "Thread-1", 1)
thread2 = rec(2, "Thread-2", 2)

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
    
# Start new Threads
thread1.start()
time.sleep(0.01)
thread2.start()

# Add threads to thread list
threads.append(thread1)
threads.append(thread2)

# Wait for all threads to complete
for t in threads:
    t.join()

# Path to session folder
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
print(('recording duration '+f"{rec_dur:.3}"+' s'+'\n resolution :'+str(w)+'x'+str(h)+ '; fps : '+str(fps)))
print('number of frames:', len(timestamps))

# Show the graph of time of arrival of each frame (should be linear)
plt.plot(range(len(timestamps)),timestamps)
plt.title(('recording duration '+f"{rec_dur:.3}"+' s'+'\n resolution :'+str(w)+'x'+str(h)+ '; fps : '+str(fps)))
plt.xlabel('frame')
plt.ylabel('epoch time in seconds')
plt.savefig(pth+'/time_graph.jpg')
plt.show()

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
import json
import re
import keyboard

# getting Date and time
now = datetime.now()
today = date.today()

# initializing things
SessDir=[]
stop = Queue()
# Initializing the queues
color_image_queue, depth_frame_queue=Queue(),Queue()
param_queue=Queue()

# set the recording time
rec_time= 1000 #int(input('set the recording time in seconds'))

class recorder():
    def __init__(self):
        # Setting the parameters of the stream
        # self.h = 720  
        # self.w = 1280 
        self.h = 480 
        self.w = 640
        self.fps=30
        self.f=0
        self.files_created=False
        self.event = threading.Event()
        self.is_saving=threading.Event()
        self.task_markers=[]

        # Initializing lists of color, depth, and parameters
        self.stream_parm_list=[]

        # initializing base parameters
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        self.sensor = self.device.first_color_sensor()
        self.sensor.set_option(rs.option.exposure,100)

        self.pc = rs.pointcloud()

    def run(self):
        self.t1 = threading.Thread(target=self.readframe)
        self.t2 = threading.Thread(target=self.processing_and_saving)
        self.t3 = threading.Thread(target=self.save2)
        self.t1.start()
        self.t2.start()
        time.sleep(0.2)
        self.t3.start()
        self.t1.join()
        self.t2.join()
        self.t3.join()
        # saving the directory for saving the time graph
        self.paramFile.close()
        print('threads done')

    def processing_and_saving(self):
        # running the the second thread for calculating pointcloud and saving            
        # Setting counters
        self.counter = 1
        self.fileCounter = 1

        # Creating necessary files
        if not self.files_created:
            self.createFile(self.fileCounter)
            SessDir.append(self.sessionDir)
            self.files_created=True

        # Saving stream parameters to parameters file
        parameters=param_queue.get()
        p_packed = msgp.packb(str(parameters)+str(self.fps))
        self.paramFile.write(p_packed)
        
        while not self.event.is_set():
            thread = threading.current_thread()
            # print(f'Worker thread: name={thread.name}, idnet={threading.get_ident()}, id={threading.get_native_id()}')
            # Get color image and depth frame from queue to calculate pointcloud and save to files     
            while not color_image_queue.empty():
                # Getting the frames from queue
                color_image=color_image_queue.get()
                depth_frame=depth_frame_queue.get()

                # getting the time stamp of the depth frame
                timestamp=(rs.frame.get_frame_metadata(depth_frame,rs.frame_metadata_value.time_of_arrival))

                # Calculating the pointcloud
                try:
                    points = self.pc.calculate(depth_frame)
                    v = points.get_vertices()
                    verts = np.asanyarray(v).view(np.float32)
                    xyzpos=verts.reshape(self.h,self.w, 3)  # xyz
                    xyzpos=xyzpos.astype(np.float16)            
                except:
                    print('error in pointcloud calculation')

                while self.depthfile.closed or self.colourfile.closed:
                    pass
                self.is_saving.set()
                # Saving frames to msgpack files
                self.save_frames(color_image, xyzpos, timestamp, 
                                self.colourfile, self.depthfile, self.paramFile)
                self.is_saving.clear()
                
                # Counting frames in each msgpack
                self.counter = self.counter + 1

                # When 90 frames in one .msgpack file, open a new file
                if self.counter == 90:
                    self.fileCounter = self.fileCounter + 1
                    while self.is_saving.is_set():
                        pass
                    self.colourfile.close()
                    self.depthfile.close()
                    self.createFile(self.fileCounter)
                    self.counter = 1  
            if self.event.is_set():
                print('event set')
                self.event.set()
                break
    
    def save2(self):
        while not self.event.is_set():
            # Get color image and depth frame from queue to calculate pointcloud and save to files     
            while not color_image_queue.empty():
                # Getting the frames from queue
                color_image=color_image_queue.get()
                depth_frame=depth_frame_queue.get()

                # getting the time stamp of the depth frame
                timestamp=(rs.frame.get_frame_metadata(depth_frame,rs.frame_metadata_value.time_of_arrival))

                # Calculating the pointcloud
                try:
                    points = self.pc.calculate(depth_frame)
                    v = points.get_vertices()
                    verts = np.asanyarray(v).view(np.float32)
                    xyzpos=verts.reshape(self.h,self.w, 3)  # xyz
                    xyzpos=xyzpos.astype(np.float16)            
                except:
                    print('error in pointcloud calculation')

                while self.depthfile.closed or self.colourfile.closed:
                    pass
                self.is_saving.set()
                # Saving frames to msgpack files
                self.save_frames(color_image, xyzpos, timestamp, 
                                self.colourfile, self.depthfile, self.paramFile)
                
                self.is_saving.clear()
                
                # Counting frames in each msgpack
                self.counter = self.counter + 1

                # When 90 frames in one .msgpack file, open a new file
                if self.counter == 90:
                    self.fileCounter = self.fileCounter + 1
                    while self.is_saving.is_set():
                        pass
                    self.colourfile.close()
                    self.depthfile.close()
                    self.createFile(self.fileCounter)
                    self.counter = 1  
            if self.event.is_set():
                print('event set')
                self.event.set()
                break

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
            self.sessionName = "Session_que_" + self.tm1 + "_" + self.tm2 + "_" + str(self.rnd)+str(self.f)
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
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        # Checking if there is an RGB camera
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        # Starting the stream for depth/color cameras
        config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)

        config.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, self.fps)

        try:
            # Start streaming from file
            profile=pipeline.start(config)    

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
            start_time = time.time()
            while time.time() - start_time < rec_time:
                print(f"{rec_time-(time.time() - start_time):.0f}" , end = '\r')
                # Checking if there are more frames
                frame_present, frameset = pipeline.try_wait_for_frames()
                    
                #End loop once video finishes
                if not frame_present:
                    break

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

                # Signal that thread1 is running
                stop.put(False)
                
                # limit size of queues to 10
                while color_image_queue.qsize()>10:
                    color_image_queue.get()
                    depth_frame_queue.get()

                # putting the images in the queues
                color_image_queue.put(color_image)
                depth_frame_queue.put(aligned_depth_frame)

                if cv2.waitKey(1) == ord('s'):
                    task_mark=rs.frame.get_frame_metadata(aligned_depth_frame,rs.frame_metadata_value.time_of_arrival)/1000
                    self.task_markers.append(task_mark)
                    print(f"task marker {len(self.task_markers)} added at time {task_mark-now.timestamp()}")
                    cv2.putText(color_image, str(f"task marker {len(self.task_markers)} added at time {task_mark-now.timestamp()}"), (10, 460), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)

                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', color_image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                if keyboard.is_pressed('q'):
                    break
                c+=1
                
        finally:

            # Stop streaming
            for i in range(10):
                stop.put(True)
            self.event.set()
            pipeline.stop()
            cv2.destroyAllWindows()

            # Create a dictionary with keys as task1, task2, etc.
            dictionary = {f"task{i+1}": value for i, value in enumerate(self.task_markers)}

            # Write the dictionary to a JSON file
            filename = SessDir[0]+"\\task_markers.json"
            with open(filename, 'w') as file:
                json.dump(dictionary, file, indent=4)
            print(f"Dictionary written to {filename}.")

            time.sleep(1)


        
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

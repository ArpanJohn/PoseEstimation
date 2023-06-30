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
import glob
import matplotlib.pyplot as plt
from multiprocessing import freeze_support , Process, Manager
from queue import Queue

# getting Date and time
now = datetime.now()
today = date.today()

class rec():
    def __init__(self, processID, name):
        # Setting the parameters of the stream
        self.h=480
        self.w=640
        self.fps=30
        self.windowscale=1

        # Getting processID
        self.processID=processID
        self.f=0
        self.name=name
        self.depth_frame_list=[]
   
    def process_1(self):
        # running the first process for recording 
        print ("Starting p1")
        self.readframe()

    def process_2(self):
        self.stop.put(False)
        # running the the second process for calculating pointcloud and saving
        print("Starting p2")
        # saving resolution as a list to add to parameters file
        self.xy = [self.w,self.h]
        
        # Setting counters
        self.counter = 1
        self.fileCounter = 1

        # Creating necessary files
        self.createFile(self.fileCounter)

        # Saving stream parameters to parameters file
        if self.param_queue.qsize() == 0:
            self.param_queue.put(101)
        parameters=self.param_queue.get()

        p_packed = msgp.packb(str(parameters)+str(self.fps))
        self.paramFile.write(p_packed)
    
        while not self.stop.get():
            # Initializing the point cloud object
            pc = rs.pointcloud()

            # Get color image and depth frame from queue to calculate pointcloud and save to files     
            while not self.color_image_queue.empty():
                # print number of items in queue
                print('Queue size p2 : ', self.color_image_queue.qsize())
                # Getting the frames from queue
                color_image=self.color_image_queue.get()
                print('in p2',len(self.depth_frame_list))
                try:
                    depth_frame=self.depth_frame_list.pop(0)

                    # getting the time stamp of the depth frame
                    timestamp=(rs.frame.get_frame_metadata(depth_frame,rs.frame_metadata_value.time_of_arrival))

                    # Calculating the pointcloud
                    try:
                        points = pc.calculate(depth_frame)
                        v = points.get_vertices()
                        verts = np.asanyarray(v).view(np.float32)
                        xyzpos=verts.reshape(self.h,self.w, 3)  # xyz
                        xyzpos=xyzpos.astype(np.float16)            
                    except:
                        print('error in pointcloud calculation')

                    # Saving frames to msgpack files
                    rec.save_frames(color_image, xyzpos, timestamp, 
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
                except:
                    pass

        # saving the directory for saving the time graph
        self.paramFile.close()
        print ("Exiting " + self.name)

    def run(self):

        with Manager() as man:
            # Initializing the queues
            self.color_image_queue = man.Queue()
            # depth_frame_queue=Queue()
            self.param_queue=man.Queue()
            self.stop = man.Queue()  
            self.SessDir=man.Queue()
            # Create two process objects
            processes = []
            p1 = Process(target=self.process_1)
            p2 = Process(target=self.process_2)

            

            # starting them
            p1.start()
            time.sleep(1)
            p2.start()


            # appending them
            processes.append(p1)
            processes.append(p2)

            for p in processes:
                p.join()    

            pth = self.SessDir.get()

            # Getting the parameter file
            targetPattern_param = f"{pth}\\PARAM*"
            ppth = glob.glob(targetPattern_param)

            #obtaining parameters and list time_stamps
            p = open(ppth[0], "rb")
            unpacker=None
            unpacker = msgp.Unpacker(p, object_hook=mpn.decode)
            prm = []
            f=0
            ps = []
            print('unpacking param file')
            for unpacked in unpacker:
                f+=1
                if f<3:
                    ps.append(unpacked)
                    continue
                prm.append(unpacked)
            timestamps=prm

            # Getting the parameters of the recording
            w=ps[0][0]
            h=ps[0][1]
            fps=ps[-1]
            rec_dur=timestamps[-1]-timestamps[0]


            # Print the parameters of the recording
            print(('recording duration '+f"{rec_dur:.3}"+' s'+'\n resolution :'+str(w)+'x'+str(h)+ '; fps : '+str(fps)))

            # Show the graph of time of arrival of each frame (should be linear)
            plt.plot(range(len(timestamps)),timestamps)
            plt.title(('recording duration '+f"{rec_dur:.3}"+' s'+'\n resolution :'+str(w)+'x'+str(h)+ '; fps : '+str(fps)))
            # plt.legend(['d','c'])
            plt.xlabel('frame')
            plt.ylabel('epoch time in seconds')
            plt.savefig(pth+'/time_graph.jpg')
            plt.show()

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
        self.depthfilename = self.sessionDir + "/" + "POINT" + "_" + self.tm + "_" + str(
            self.rnd) + "_" + str(fileCounter) + ".msgpack"
        self.colourfilename = self.sessionDir + "/" + "COLOUR" + "_" + self.tm + "_" + str(
            self.rnd) + "_" + str(fileCounter) + ".msgpack"

        # Opening depth and colour file for writing
        self.depthfile = open(self.depthfilename, 'wb')
        self.colourfile = open(self.colourfilename, 'wb')
        self.SessDir.put(self.sessionDir)

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
            self.param_queue.put('depth intrinsics:'+intr) 

            # initializing alignment
            align_to = rs.stream.color
            align = rs.align(align_to)

            # Number of frames 
            c=0
            
            while True:
                # Checking if there are more frames
                frame_present, frameset = pipeline.try_wait_for_frames()
                    
                #End loop once video finishes
                if not frame_present:
                    break

                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()

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

                # Signal that process1 is running
                self.stop.put(False)

                # limit size of queues to 10
                while self.color_image_queue.qsize()>10 or len(self.depth_frame_list)>10:
                    # print('binning queue' , end = '\r')
                    # print(type(color_image_queue.get()))
                    bin=self.color_image_queue.get()
                    bin=self.depth_frame_list.pop(0)
                    # bin=param_queue.get()
                    bin=None

                # print number of items in queue
                # print('Queue size p1 : ', color_image_queue.qsize(), end = '\r')
                # putting the images in the queues
                self.color_image_queue.put(color_image)
                self.depth_frame_list.append(aligned_depth_frame)
                print('in p1',len(self.depth_frame_list))

                # resizing for display
                color_image=cv2.resize(color_image, (int(self.w*self.windowscale),int(self.h*self.windowscale)))
                    
                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', color_image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    self.stop.put(True)
                    break
                c+=1
                
        finally:

            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    freeze_support()
    Rec = rec(1,'process 1')
    Rec.run()   

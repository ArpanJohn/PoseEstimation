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
import pickle
from numpy import random
import sys

now = datetime.now()
today = date.today()
dtime=[]
ctime=[]
# Setting the parameters of the stream
h=480
w=640
fps=30
windowscale=1

# choose video codec
xvid=True # compressed
rgba=True # Uncompressed 32-bit RGBA
mjpg=True # Motion JPEG
iyuv=True # 
mp4v=True # MPEG-4 Video
b48r=True # Uncompressed 48-bit RGB

class rec:
    def __init__(self, *args, **kwargs):
        self.f=0
        # super(rec, self).__init__(*args, **kwargs)
        worker = self.readframe()


    def createFile(self, fileCounter):
        global fps,w,h
        self.pFileName = 'test'
        self.tm1 = today.strftime("%d-%m-%y")
        self.tm2 = now.strftime("%H-%M-%S")
        self.tM = self.tm1 + " " + self.tm2
        self.tm = self.tm1 + "_" + self.tm2
        self.savingDir = r'C:\Users\arpan\OneDrive\Documents\internship\rec_program\savdir'
        self.temp_dir = r'C:\Users\arpan\OneDrive\Documents\internship\rec_program\temp_dir'
        self.f=self.f+1
        if fileCounter == 1:
            self.rnd = random.randint(999)
            self.sessionName = "Session " + self.tm1 + "_" + self.tm2 + "_" + str(self.rnd)+str(self.f)
            self.sessionDir = os.path.join(self.savingDir, self.sessionName)
            self.temp_save = os.path.join(self.temp_dir, self.sessionName)
            os.mkdir(self.sessionDir)

            self.parmsFileName = self.sessionDir + "/" + "PARAMS" + "_" + self.tm + "_" + str(
                self.rnd) + ".msgpack"

            self.paramFile = open(self.parmsFileName, 'wb')
            # Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
            if xvid:
                self.vid_filename = self.sessionDir + "/" + "Videoxvid.avi"    
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.cv_writer_xvid = cv2.VideoWriter(self.vid_filename, fourcc, fps,(w,h))
            if rgba:
                self.vid_filename = self.sessionDir + "/" + "Videorgba.avi"    
                fourcc = cv2.VideoWriter_fourcc(*'RGBA')
                self.cv_writer_rgba = cv2.VideoWriter(self.vid_filename, fourcc, fps,(w,h))
            if mjpg:
                self.vid_filename = self.sessionDir + "/" + "Videomjpg.avi"    
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.cv_writer_mjpg = cv2.VideoWriter(self.vid_filename, fourcc, fps,(w,h))
            if iyuv:
                self.vid_filename = self.sessionDir + "/" + "Videoiyuv.avi"    
                fourcc = cv2.VideoWriter_fourcc(*'IYUV')
                self.cv_writer_iyuv = cv2.VideoWriter(self.vid_filename, fourcc, fps,(w,h))
            if mp4v:
                self.vid_filename = self.sessionDir + "/" + "Videomp4v.mp4"    
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.cv_writer_mp4v = cv2.VideoWriter(self.vid_filename, fourcc, fps,(w,h))
            if b48r:
                self.vid_filename = self.sessionDir + "/" + "Videob48r.mp4"    
                fourcc = cv2.VideoWriter_fourcc(*'b48r')
                self.cv_writer_b48r = cv2.VideoWriter(self.vid_filename, fourcc, fps,(w,h))

        self.commonName = self.pFileName + " " + self.tM + " " + str(self.rnd)
        self.depthfilename = self.sessionDir + "/" + "DEPTH" + "_" + self.tm + "_" + str(
            self.rnd) + "_" + str(fileCounter) + ".msgpack"
        self.colourfilename = self.sessionDir + "/" + "COLOUR" + "_" + self.tm + "_" + str(
            self.rnd) + "_" + str(fileCounter) + ".msgpack"
        self.depthfile = open(self.depthfilename, 'wb')
        self.colourfile = open(self.colourfilename, 'wb')

        print(f"creating files {fileCounter}")

            
    def save_frames(colorImg, depthImg, milliseconds, colorFile, depthFile, paramsfile):
        d_packed = msgp.packb(depthImg, default=mpn.encode)
        depthFile.write(d_packed)

        c_packed = msgp.packb(colorImg, default=mpn.encode)
        colorFile.write(c_packed)

        prm = [milliseconds]
        p_packed = msgp.packb(prm)
        paramsfile.write(p_packed)

    def readframe(self):
        global fps

        self.xy = [w,h]

        self.counter = 1
        self.fileCounter = 1

        self.createFile(self.fileCounter)
        p_packed = msgp.packb(self.xy)
        self.paramFile.write(p_packed)
        p_packed = msgp.packb(2)
        self.paramFile.write(p_packed)

        new_frame_time = 0
        prev_frame_time = 0
        
        # Creating pointcloud object
        pc = rs.pointcloud()
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)

        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

        try:
            # Start streaming from file
            profile=pipeline.start(config)
            
            # Create colorizer object
            colorizer = rs.colorizer()

            align_to = rs.stream.color
            align = rs.align(align_to)

            # Initializing the list of timestamps
            timestamps=[]

            # Number of frames 
            c=0

            while True:
                frame_present, frameset = pipeline.try_wait_for_frames()
                    
                #End loop once video finishes
                if not frame_present:
                    break

                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue                                
                
                # get timestamp of frame
                timestamps.append((rs.frame.get_frame_metadata(depth_frame,rs.frame_metadata_value.time_of_arrival)))

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame() 
                color_frame = aligned_frames.get_color_frame()

                # Convert images to numpy arrays
                depth_image = np.asanyarray(aligned_depth_frame.get_data()) 
                color_image = np.asanyarray(color_frame.get_data()) 

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Finding the dimensions of the depth and colour image
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

                # resizing for display
                color_image=cv2.resize(color_image, (int(w*windowscale),int(h*windowscale)))
                depth_colormap=cv2.resize(depth_colormap, (int(w*windowscale),int(h*windowscale)))
                images =np.hstack((color_image, depth_colormap))
                if self.counter == 90:
                    self.fileCounter = self.fileCounter + 1
                    self.colourfile.close()
                    self.depthfile.close()
                    self.createFile(self.fileCounter)
                    self.counter = 1

                new_frame_time = time.time()

                # Calculating the point cloud
                mapped_frame = color_frame
                pc.map_to(mapped_frame)  
                try:
                    points = pc.calculate(aligned_depth_frame)
                    v = points.get_vertices()
                    verts = np.asanyarray(v).view(np.float32)
                    xyzpos=verts.reshape(h,w, 3)  # xyz
                    xyzpos=xyzpos.astype(np.float16)
                    # Saving the 3D position information in a file
                    with open('posout.npy', 'ab') as f:
                        np.save(f,xyzpos)   
                except:
                    print(type(v))

                rec.save_frames(color_image, xyzpos, (rs.frame.get_frame_metadata(depth_frame,rs.frame_metadata_value.time_of_arrival)), self.colourfile, self.depthfile, self.paramFile)
                
                if xvid:
                    self.cv_writer_xvid.write(color_image)
                if rgba:
                    self.cv_writer_rgba.write(color_image)
                if mjpg:
                    self.cv_writer_mjpg.write(color_image)
                if iyuv:
                    self.cv_writer_iyuv.write(color_image)
                if mp4v:
                    self.cv_writer_mp4v.write(color_image)
                if b48r:
                    self.cv_writer_b48r.write(color_image)

                self.counter = self.counter + 1
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                c+=1
                dtime.append(rs.frame.get_frame_metadata(depth_frame,rs.frame_metadata_value.time_of_arrival))
                ctime.append(rs.frame.get_frame_metadata(color_frame,rs.frame_metadata_value.time_of_arrival))
                
        finally:

            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()


run=rec()

import matplotlib.pyplot as plt

print('dtime duration',(dtime[-1]-dtime[0])/1000,'s')
print('ctime duration',(ctime[-1]-ctime[0])/1000,'s')


plt.plot(range(len(dtime)),dtime)
plt.plot(range(len(dtime)),ctime)
plt.legend(['d','c'])
plt.show()



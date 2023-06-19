# Importing necessary libraries
import numpy as np
import cv2
import msgpack as msgp
import msgpack_numpy as mpn
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# choose whether to play video 
playback = True

# choose video codec
xvid = True  # compressed
rgba = False  # Uncompressed 32-bit RGBA
mjpg = True  # Motion JPEG
iyuv = True 
mp4v = True  # MPEG-4 Video
b48r = True  # Uncompressed 48-bit RGB

# Setting the parameters of the stream
h = 480  # 720
w = 640  # 1280
fps = 30
windowscale = 1

# Get the last session
import os

# Path to session folder
pth = r"C:\Users\arpan\OneDrive\Documents\internship\rec_program\savdir\Session 19-06-23_12-14-21_2341"

lst = os.listdir(pth)
vid_name = lst[-1]

# Getting the DEPTH files
targetPattern = f"{pth}\\DEPTH*"
campth = glob.glob(targetPattern)

# Getting the PARAM files
targetPattern_param = f"{pth}\\PARAM*"
ppth = glob.glob(targetPattern_param)

# Getting the COLOR files
targetPattern_colour = f"{pth}\\COLOUR*"
cpth = glob.glob(targetPattern_colour)

# Define writer with defined parameters and suitable output filename
if xvid:
    vid_filename = pth + "/" + "Videoxvid.avi"    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cv_writer_xvid = cv2.VideoWriter(vid_filename, fourcc, fps, (w, h))
if rgba:
    vid_filename = pth + "/" + "Videorgba.avi"    
    fourcc = cv2.VideoWriter_fourcc(*'RGBA')
    cv_writer_rgba = cv2.VideoWriter(vid_filename, fourcc, fps, (w, h))
if mjpg:
    vid_filename = pth + "/" + "Videomjpg.avi"    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cv_writer_mjpg = cv2.VideoWriter(vid_filename, fourcc, fps, (w, h))
if iyuv:
    vid_filename = pth + "/" + "Videoiyuv.avi"    
    fourcc = cv2.VideoWriter_fourcc(*'IYUV')
    cv_writer_iyuv = cv2.VideoWriter(vid_filename, fourcc, fps, (w, h))
if mp4v:
    vid_filename = pth + "/" + "Videomp4v.avi"    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cv_writer_mp4v = cv2.VideoWriter(vid_filename, fourcc, fps, (w, h))
if b48r:
    vid_filename = pth + "/" + "Videob48r.avi"    
    fourcc = cv2.VideoWriter_fourcc(*'b48r')
    cv_writer_b48r = cv2.VideoWriter(vid_filename, fourcc, fps, (w, h))

c = 0

# Iterate over each file path in 'cpth'
for i in cpth:
    print(i)

    # Open the file in binary mode
    col_file = open(i, "rb")

    # Create an unpacker object with custom decoding using msgpack_numpy
    unpacker = msgp.Unpacker(col_file, object_hook=mpn.decode)

    # Iterate over each unpacked data in the unpacker
    for unpacked in unpacker:
        c += 1

        # Convert unpacked data to color image
        color_image = unpacked

        # Write the color image to respective video writers based on selected codecs
        if xvid:
            cv_writer_xvid.write(color_image)
        if rgba:
            cv_writer_rgba.write(color_image)
        if mjpg:
            cv_writer_mjpg.write(color_image)
        if iyuv:
            cv_writer_iyuv.write(color_image)
        if mp4v:
            cv_writer_mp4v.write(color_image)
        if b48r:
            cv_writer_b48r.write(color_image)

        if playback:
            # Display the resulting image
            cv2.imshow("Pose Landmarks", color_image)

        # Press 'q' key to break the loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
        try:
            if unpacked == -1:
                cv2.destroyAllWindows()
                break
        except:
            pass

    # Close the file after processing
    col_file.close()

cv2.destroyAllWindows()


#adding something here

#atonve

#twice
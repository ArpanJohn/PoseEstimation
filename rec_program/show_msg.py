# Importing necessary libraries
import numpy as np
import cv2
import msgpack as msgp
import msgpack_numpy as mpn
import glob
from natsort import natsorted
import re

# Setting the parameters of the stream
fps = 30
windowscale = 1


# Path to session folder
pth = r"C:\Users\arpan\OneDrive\Documents\internship\rec_program\savdir\Session_27-06-23_08-29-51_561"

# Getting the COLOR files
targetPattern_colour = f"{pth}\\COLOUR*"
cpth = glob.glob(targetPattern_colour)

# Gettin the DEPTH files
targetPattern = f"{pth}\\POINT*"
campth = glob.glob(targetPattern)

# Getting the parameter file
targetPattern_param = f"{pth}\\PARAM*"
ppth = glob.glob(targetPattern_param)

# sorting
cpth=natsorted(cpth)

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

for unpacked in unpacker:
    timestamps.append(unpacked)

rec_dur=timestamps[-1]-timestamps[0]

# Print the parameters of the recording
print(('recording duration '+f"{rec_dur:.3}"+' s'+'\nresolution :'+str(w)+'x'+str(h)+ '; fps : '+str(fps)))

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
        color_image = cv2.flip(unpacked,1)

        # Display the resulting image
        cv2.imshow("extracted image", color_image)

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

# Iterate over each file path in 'cpth'
for i in campth:
    print(i)

    # Open the file in binary mode
    col_file = open(i, "rb")

    # Create an unpacker object with custom decoding using msgpack_numpy
    unpacker = msgp.Unpacker(col_file, object_hook=mpn.decode)

    # Iterate over each unpacked data in the unpacker
    for unpacked in unpacker:
        c += 1
        unpacked=np.asanyarray(unpacked)
        unpacked=unpacked.reshape(h,w,3)
        unpacked=unpacked[:,:,2]
        # Convert unpacked data to color image
        color_image = cv2.flip(unpacked,1)

        # Display the resulting image
        cv2.imshow("extracted image", color_image)

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


import math
import numpy as np
from math import sqrt
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def pvt(x,t,ylabel='Angle (degrees)'):
    y=[]
    for i in x:
        y.append(i)
    w = savgol_filter(y,  int(len(y)/20),3)
    plt.plot(t,w)
    plt.ylabel(ylabel)
    plt.xlabel('time(s)')    

def frame_con(point,rotmat,org):
    point=np.array(point)
    point=point.reshape(3,1)
    point=point-org 
    point=np.matmul(rotmat.T,point)
    return point

def RMSE(y_true, y_pred,time_true,time_pred):
    from scipy.signal import savgol_filter
    '''
    y_true,y_pred : list/listlike containing the true and predicted values respectively
    time_true,time_pred : list/listlike containing the time values
    returns : RMSE value
    '''
    yp2,tp2,yt2,tt2=[],[],[],[]

    for i in range(len(y_pred)):
        if not math.isnan(y_pred[i]):
            yp2.append(y_pred[i])
            tp2.append(time_pred[i])
    
    for i in range(len(y_true)):
        if not math.isnan(y_true[i]):
            yt2.append(y_true[i])
            tt2.append(time_true[i])
   
    y_pred=yp2
    time_pred=tp2
    y_true=yt2
    time_true=tt2

    y_pred=savgol_filter(y_pred,  int(len(y_pred)/20),3)
    y_true=savgol_filter(y_true,  int(len(y_true)/20),3)
    y_true_dic={}
    y_true_shortened=[]
    y_true_dic = {time_true[i]: y_true[i] for i in range(len(y_true))}


    y_true=list(y_true)
    y_pred=list(y_pred)
    time_true=list(time_true)
    time_pred=list(time_pred)
    #removing excess values and making pred and true the same size
    while(time_pred[0]<time_true[0]):
        time_pred.pop(0)
        y_pred.pop(0)
    while(time_pred[-1]>time_true[-1]):
        time_pred.pop(-1)
        y_pred.pop(-1)
    while(time_pred[0]>time_true[0]):
        time_true.pop(0)
        y_true.pop(0)
    while(time_pred[-1]>time_true[-1]):
        time_true.pop(-1)
        y_true.pop(-1)

    for i in range(1,len(time_pred)):
        mean_calc=[]
        n=0
        for key,value in y_true_dic.items():
            if time_pred[i]>=key>=time_pred[i-1]:
                mean_calc.append(value)
                n+=1
        if n!=0:
            y_true_shortened.append(sum(mean_calc)/n)      
    while (len(y_true_shortened)<len(y_pred)):
        y_pred.pop(0)
    while (len(y_true_shortened)>len(y_pred)):
        y_true_shortened.pop(0)

    return np.sqrt(np.mean((np.array(y_true_shortened) - np.array(y_pred)) ** 2)  )

def angle3point(a,b,c):
    import mpmath
    '''
    a,b,c : 1D lists/like representing the points
    return : angle in degrees
    '''
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    
    ba = a - b
    bc = c - b


    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def angle2vec(a,b):
    '''
    a,b : 1D lists/like representing the vectors
    return : angle in degrees
    '''
    a=np.array(a)
    b=np.array(b)
    
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def project_onto_plane(point, plane_points):
    '''
    Projects a point onto a plane.
    point: a list containing a 3D point
    plane_points: a list containing 3 3D points on the plane
    returns : projection of the point onto the plane
    '''
    # Convert the points to numpy arrays
    point = np.array(point)
    plane_points = np.array(plane_points)
    # print(point,'\n',plane_points)

    # Calculate the plane's normal vector
    v1 = plane_points[1] - plane_points[0]
    v2 = plane_points[2] - plane_points[0]
    # print(v1,'\n',v2)

    #calculate unit normal
    normal = np.cross(v1, v2)
    unitnorm=normal/np.linalg.norm(normal)

    #point projection onto normal
    pp=np.dot(unitnorm,point) * unitnorm

    # Calculate the projection of the point onto the plane
    projection = point - pp
    
    return projection.tolist()

def find_orthogonal_points(point1, point2, point3,vec1):
    # Calculate the normal vector of the given plane
    normal_vector = np.cross(point2 - point1, point3 - point1)

def sag_plane(chestplanepoints):

    ''' 
    to calculate the saggital plane
    chestplanepoints: a list containing 3 3D points on the chest
    returns org, cross point, tr point
    '''
    pl=np.array(chestplanepoints)
    org=(pl[1]-pl[2])/2
    norm=pl[1]-org
    v1=pl[0]-org # straight down
    v2=np.cross(norm,v1)
    
    # print(org,'\n',norm,'\n',v,'\n',cross,'\n')
    return[org,org+v2,org+v1]

def midpoint(p1,p2):

    '''
    calculates the midpoint of two points
    p1,p2 : lists/like representing the points
    returns : midpoint
    '''
    mid=[]
    for i in range(len(p1)):
        mid.append((p1[i]+p2[i])/2)
    return mid

def read_df_csv(filename, offset=2):
    """
    this function reads the csv file from motion capture system
    and makes it into a dataframe and returns only useful information

    filename: input path to csv file from motive
    offset:to ignore the first two columns with time and frames generally

    """
    import pandas as pd
    import datetime
    # offset = 2 #first two columns with frame_no and time

    pth = filename
    raw = pd.read_csv(pth)
    cols_list = raw.columns     # first row which contains capture start time
    inx = [i for i, x in enumerate(cols_list) if x == "Capture Start Time"]
    st_time = cols_list[inx[0] + 1]
    st_time = datetime.datetime.strptime(st_time, "%Y-%m-%d %I.%M.%S.%f %p")  # returns datetime object

    mr_inx = pd.read_csv(pth, skiprows=3)
    markers_raw = mr_inx.columns
    marker_offset = offset  # for ignoring time and frame cols
    markers_raw = markers_raw[marker_offset:]
    col_names = []
    for i in range(0, len(markers_raw), 3):
        col_names.append(markers_raw[i].split(":")[1])

    df_headers = ["frame", "seconds"]

    for id, i in enumerate(col_names):
        if not i.islower():
            col_names[id] = i
    
    for i in col_names:
        df_headers.append(i + "_x")
        df_headers.append(i + "_y")
        df_headers.append(i + "_z")
    mo_data = pd.read_csv(pth, skiprows=6)
    # mo_data = mo_data.rename(mo_data.columns, df_headers)
    mo_data.columns = df_headers

    mo_data.drop(['frame'],axis=1,inplace=True)

    return mo_data, st_time

def perp_bisector(p1, p2):
    # Find the midpoint
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2


    # Find the slope of the line connecting the two points
    if p2[0] - p1[0] == 0:
        slope = float('inf')
    else:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])


    # Find the slope of the perpendicular bisector
    if slope == 0:
        perp_slope = float('inf')
    elif slope == float('inf'):
        perp_slope = 0
    else:
        perp_slope = -1 / slope


    # Find two points on the perpendicular bisector
    if perp_slope == float('inf'):
        x3 = mid_x
        y3 = mid_y + 1
        x4 = mid_x
        y4 = mid_y - 1
    else:
        x3 = mid_x + 1
        y3 = perp_slope * (x3 - mid_x) + mid_y
        x4 = mid_x - 1
        y4 = perp_slope * (x4 - mid_x) + mid_y

    return [x3, y3], [x4, y4] 

def draw_line(image, point1, point2, color=(0, 0, 255), thickness=2):
    import cv2
    # Create a blank image canvas to draw the line
    canvas = np.zeros_like(image)

    # Convert the points to tuples of integers
    pt1 = (int(point1[0]), int(point1[1]))
    pt2 = (int(point2[0]), int(point2[1]))

    # Draw the line on the canvas
    cv2.line(canvas, pt1, pt2, color, thickness)

    # Overlay the line on the original image
    result = cv2.addWeighted(image, 1, canvas, 1, 0)

    return result

def generate_line(point1, point2,r=20):
    import math
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the slope and y-intercept of the line
    if x2 - x1 == 0:
        slope = float('inf')
        y_intercept = x1
    else:
        slope = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - slope * x1

    # Generate the list of points
    centers = []
    if slope == float('inf'):
        for y in range(int(min(y1, y2)), int(max(y1, y2)) , 1):
            centers.append((x1, y))
    else:
        for x in range(int(min(x1, x2)),int(max(x1, x2)) , 1):
            y = int(slope * x + y_intercept)
            centers.append((x, y))

    points = []
    for i in centers:
        x0,y0=i
        for x in range(x0 - r, x0 + r + 1):
            for y in range(y0 - r, y0 + r + 1):
                if (x - x0) ** 2 + (y - y0) ** 2 <= r ** 2:
                    points.append((x, y))

    return points

def color_pixels(image, points, color=(0,0,255)):
    import numpy as np
    # Create a copy of the image to avoid modifying the original
    colored_image = np.copy(image)

    # Change the color of each pixel in the set of points
    for point in points:
        x, y = point
        if x>1279 or y >719 or x<0 or y<0:
            continue
        colored_image[y, x] = color
    
    return colored_image

def points_in_disc(p, r=30):
    x, y = p
    points = []
    for i in range(int(x-r), int(x+r)+1):
        for j in range(int(y-r), int(y+r)+1):
            if sqrt((x-i)**2 + (y-j)**2) <= r:
                points.append((i,j))
    return points

def calc_col_derivatives(df, column_name):
    """
    Calculates the derivatives of elements in a pandas DataFrame column using the finite difference method.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column.
        column_name (str): The name of the column to calculate derivatives for.

    Returns:
        pandas.Series: A new Series containing the derivatives.
    """
    column = df[column_name]  # Extract the column as a Series
    derivatives = column.diff()  # Calculate the finite differences
    return derivatives

import cv2

def point_in_quad(point, quadrilateral):
    from matplotlib.path import Path

    vertices = [(p[0], p[1]) for p in quadrilateral]
    path = Path(vertices)
    return path.contains_point(point)

def draw_box(image, point1, point2, color=(0, 255, 0), dis=30):
    """
    Draw a box around two points using OpenCV.

    :param image: Image on which to draw the box
    :param point1: Tuple of x and y coordinates of the first point
    :param point2: Tuple of x and y coordinates of the second point
    :param color: Tuple of BGR values for the color of the box
    :param thickness: Thickness of the lines of the box
    """
    x1, y1 = point1
    x2, y2 = point2

    angle = math.atan2(y2 - y1, x2 - x1)

    # Calculate the new points
    x1 = x1 - dis * math.cos(angle)
    y1 = y1 - dis * math.sin(angle)
    x2 = x2 + dis * math.cos(angle)
    y2 = y2 + dis * math.sin(angle)

    # Calculate the slope of the line
    slope = (y2 - y1) / (x2 - x1)

    # Calculate the slope of the perpendicular line
    perp_slope = -1 / slope

    # Calculate the angle of the perpendicular line
    angle = math.atan(perp_slope)

    # Calculate the new points
    bp1=(int(x1 + dis * math.cos(angle)),int( y1 + dis * math.sin(angle)))
    bp2=(int(x1 - dis * math.cos(angle)),int( y1 - dis * math.sin(angle)))
    bp3=(int(x2 - dis * math.cos(angle)),int( y2 - dis * math.sin(angle)))
    bp4=(int(x2 + dis * math.cos(angle)),int( y2 + dis * math.sin(angle)))

    try:
        cv2.line(image,bp1,bp2 , color, 2)
        cv2.line(image,bp2,bp3 , color, 2)
        cv2.line(image,bp3,bp4 , color, 2)
        cv2.line(image,bp4,bp1 , color, 2)
    except:
        pass

    return bp1,bp2,bp3,bp4

def find_orthogonal_frame(v1, v2):
    """
    v1:upper arm vector
    v2:lower arm vector
    """
    # Normalize the input vectors
    vy = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Compute the cross product to find the orthogonal vector
    vx = np.cross(v1, v2)
    vx=vx/np.linalg.norm(vx)
    
    # Normalize the orthogonal vector
    vy = vy / np.linalg.norm(vy)

    vz=np.cross(vx,vy)
    vz=vz/np.linalg.norm(vz)

    return np.array([vx,vy,vz])

def find_tr_frame(v1, v2,v3):
    """
    v1 : Right Shoulder
    v2 : Left Shoulder
    v3 : Trunk
    """
    # midpoint
    m=midpoint(v1,v2)
    v1=np.array(v1)
    v2=np.array(v2)
    v3=np.array(v3)
    m=np.array(m)
    
    vy=v3-m
    vy = vy / np.linalg.norm(vy)

    vz=np.cross(vy,v2)
    vz=vz/np.linalg.norm(vz)

    vx=np.cross(vz,vy)
    vx=vx/np.linalg.norm(vx)

    return np.array([vx,vy,vz])

def find_rotation_matrix(frame1_vectors, frame2_vectors):
    # Check if the number of vectors is not equal to 3
    if len(frame1_vectors) != 3 or len(frame2_vectors) != 3:
        raise ValueError("Invalid number of vectors. Expected 3 vectors for each frame.")
    
    # Normalize vectors
    for i in range(3):
        frame1_vectors[i]=frame1_vectors[i]/np.linalg.norm(frame1_vectors[i])
        frame2_vectors[i]=frame2_vectors[i]/np.linalg.norm(frame2_vectors[i])
    
    # Create the rotation matrix by concatenating the unit vectors
    frame1_matrix = np.column_stack(frame1_vectors)
    frame2_matrix = np.column_stack(frame2_vectors)
    
    for matrix in [frame1_matrix,frame2_matrix]:
        if not np.allclose(matrix.T, np.linalg.inv(matrix)) and not np.isclose(np.linalg.det(matrix), 1):
            print(matrix, 'is not valid orthonormal matrix')    
            return
    
    # Calculate the rotation matrix that transforms frame1 to frame2
    rotation_matrix = np.matmul(frame2_matrix, frame1_matrix.T)
    
    return rotation_matrix

def rotation_angles(matrix):
    """
    input
        matrix = 3x3 rotation matrix (numpy array)
        order is yzx, 
    output
        theta1, theta2, theta3 = rotation angles in rotation order
    """
    # matrix=np.array(matrix)
    # matrix=matrix.T
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    # xzy
    theta1 = np.arctan(r32 / r22)
    theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
    theta3 = np.arctan(r13 / r11)

    #yzx
    # theta1 = np.arctan(-r31 / r11)
    # theta2 = np.arctan(r21 * np.cos(theta1) / r11)
    # theta3 = np.arctan(-r23 / r22)


    theta1 = -theta1 * 180 / np.pi
    theta2 = -theta2 * 180 / np.pi
    theta3 = -theta3 * 180 / np.pi

    return [theta1, theta2, theta3]






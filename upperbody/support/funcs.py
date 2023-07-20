import math
import numpy as np
from math import sqrt
import pandas as pd
import cv2

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

    try:
        y_pred=savgol_filter(y_pred,  int(len(y_pred)/20),3)
    except:
        y_pred=savgol_filter(y_pred,  int(len(y_pred)/20)+1,3)

    y_true=savgol_filter(y_true,  5,3)
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

    return np.sqrt(np.mean((np.array(y_true_shortened) - np.array(y_pred)) ** 2)) , abs(np.max((np.array(y_true_shortened) - np.array(y_pred))))

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

def point_in_quad(point, quadrilateral):
    from matplotlib.path import Path

    vertices = [(p[0], p[1]) for p in quadrilateral]
    path = Path(vertices)
    return path.contains_point(point)

def draw_box(image, point1, point2, color=(0, 255, 0), dis=20):
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
    x1 = x1 - (dis + 5) * math.cos(angle)
    y1 = y1 - dis * math.sin(angle)
    x2 = x2 + (dis + 5) * math.cos(angle)
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

    theta1_neg = False
    theta2_neg = False
    theta3_neg = False

    if r32 < 0:
        theta1_neg = True
    if r12 > 0: 
        theta2_neg = True
    if r13 < 0:
        theta3_neg = True 

    if not theta1_neg:
        if np.rad2deg(np.arctan(r32 / r22)) < 0:
            theta1=(180+np.rad2deg(np.arctan(r32 / r22)))
        else:
            theta1=(np.rad2deg(np.arctan(r32 / r22)))
    else:
        if np.rad2deg(np.arctan(r32 / r22)) < 0:
            theta1=(np.rad2deg(np.arctan(r32 / r22)))
        else:
            theta1=(np.rad2deg(np.arctan(r32 / r22))-180)
    
    if not theta2_neg:
        if np.rad2deg(np.arctan((-r12 * np.cos(np.deg2rad(theta1)) / r22))) < 0:
            theta2=(180+np.rad2deg(np.arctan(-r12 * np.cos(np.deg2rad(theta1)) / r22)))
        else:
            theta2=(np.rad2deg(np.arctan((-r12 * np.cos(np.deg2rad(theta1)) / r22))))
    else:
        if np.rad2deg(np.arctan((-r12 * np.cos(np.deg2rad(theta1)) / r22))) < 0:
            theta2=np.rad2deg(np.arctan(-r12 * np.cos(np.deg2rad(theta1)) / r22))
        else:
            theta2=(np.rad2deg(np.arctan(-r12 * np.cos(np.deg2rad(theta1)) / r22))-180)
    
    if not theta3_neg:
        if np.rad2deg(np.arctan((r13 / r11))) < 0:
            theta3=(180+np.rad2deg(np.arctan((r13 / r11))))
        else:
            theta3=(np.rad2deg(np.arctan((r13 / r11))))
    else:
        if np.rad2deg(np.arctan((r13 / r11))) < 0:
            theta3=np.rad2deg(np.arctan((r13 / r11)))
        else:
            theta3=(np.rad2deg(np.arctan((r13 / r11)))-180)
            
    theta1 = -theta1
    theta2 = -theta2
    theta3 = -theta3

    # # xzy
    # theta1 = np.arctan(r32 / r22)
    # theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
    # theta3 = np.arctan(r13 / r11)

    # theta1 = -theta1 * 180 / np.pi
    # theta2 = -theta2 * 180 / np.pi
    # theta3 = -theta3 * 180 / np.pi

    return [theta1, theta2, theta3]


def errors(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) and maximum error between two lists.

    Parameters:
    - y_true (list): List of true values.
    - y_pred (list): List of predicted values.

    Returns:
    - rmse (float): Root Mean Squared Error.
    - max_error (float): Maximum error.

    Raises:
    - ValueError: If the lengths of y_true and y_pred are different.

    """

    # Check if both lists are of the same length
    if len(y_true) != len(y_pred):
        raise ValueError('Both lists must be of the same length.')

    # Convert the lists to NumPy arrays
    array1 = np.array(y_true)
    array2 = np.array(y_pred)

    mask = ~np.isnan(array1) & ~np.isnan(array2)
    new_array1 = array1[mask]
    new_array2 = array2[mask]

    # slope_mask1 = np.array(high_slope_index(new_array1,window_length=50,slope_threshold=1.5))
    # slope_mask2 = np.array(high_slope_index(new_array2,window_length=50,slope_threshold=1.5))
    # new_array1[slope_mask1] = np.mean(new_array1)
    # new_array2[slope_mask2] = np.mean(new_array2)

    # Calculate the squared difference between the two arrays
    squared_diff = (new_array1 - new_array2) ** 2

    # Calculate the mean of the squared differences
    mean_squared_diff = squared_diff.mean()

    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_diff)

    ## Calculate the absolute differences between the two arrays
    absolute_diff = np.abs(new_array1 - new_array2)

    # Find the maximum error and its index
    max_error = np.max(absolute_diff)
    max_error_index = np.argmax(absolute_diff)

    return rmse, max_error, max_error_index

def split_list_by_indexes(lst, indexes):
    result = []
    start = 0

    for index in indexes:
        result.append(lst[start:index])
        start = index

    # Add the remaining portion of the list
    result.append(lst[start:])

    return result

def high_slope_index(input_list, window_length=200, slope_threshold=0.005):
    output_list = []
    n=window_length
    for i in range(len(input_list)):
        start_idx = max(i - n, 0)
        end_idx = min(i + n + 1, len(input_list))
        surrounding = input_list[start_idx:end_idx]
        slope=np.polyfit(range(len(surrounding)),surrounding,1)[0]
        if abs(slope)>slope_threshold:
            output_list.append(True)
        else:
            output_list.append(False) 
    return output_list
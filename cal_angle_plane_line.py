def cal_angle_plane_line(line_points, plane_points):
    # Convert the points to numpy arrays
    line_points = np.array(line_points)
    plane_points = np.array(plane_points)

    # Calculate the direction vector of the line
    linevector = line_points[1] - line_points[0]

    # Calculate the plane's normal vector
    v1 = plane_points[1] - plane_points[0]
    v2 = plane_points[2] - plane_points[0]
    normal = np.cross(v1, v2)

    # Calculate unit vector of normal
    normcap=normal/np.linalg.norm(normal)

    # Calculate unit vector of linevector
    linecap=linevector/np.linalg.norm(linevector)

    # Calculate angle between unit vectors
    dot_product = np.dot(normcap, linecap)
    angle_radians = np.arccos(dot_product)

    # Convert the angle to degrees and with respect to plane
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

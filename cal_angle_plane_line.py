def cal_angle_plane_line(line_points, plane_points):
    # Convert the points to numpy arrays
    line_points = np.array(line_points)
    plane_points = np.array(plane_points)

    # Calculate the direction vector of the line
    line_vector = line_points[1] - line_points[0]

    # Calculate the plane's normal vector
    v1 = plane_points[1] - plane_points[0]
    v2 = plane_points[2] - plane_points[0]
    normal = np.cross(v1, v2)

    # Calculate the dot product between the line vector and the plane normal
    dot_product = np.dot(line_vector, normal)

    # Calculate the magnitudes of the line vector and the plane normal
    line_magnitude = np.linalg.norm(line_vector)
    normal_magnitude = np.linalg.norm(normal)

    # Calculate the cosine of the angle between the line and the plane
    cosine_angle = dot_product / (line_magnitude * normal_magnitude)

    # Calculate the angle in radians
    angle_radians = np.arccos(cosine_angle)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

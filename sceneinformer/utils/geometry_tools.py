import math
import random

import numpy as np


def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.
    
    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy

def rotate_around_point_highperf_vectorized(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.
    """
    #check if radians is a tuple:
    if type(radians) == tuple or type(origin) == tuple:
        x, y = xy[:,0], xy[:, 1]
        offset_x, offset_y = origin[0], origin[1]
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = math.cos(radians)
        sin_rad = math.sin(radians)
    else:
        x, y = xy[:,0], xy[:, 1]
        offset_x, offset_y = origin[:, 0], origin[:, 1]
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = np.cos(radians)
        sin_rad = np.sin(radians)

    xy[:,0] = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    xy[:,1] = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return xy

def normalize(points, heading, point_centre):
    # 0 - position x
    # 1 - position y
    points[:,:2] = rotate_around_point_highperf_vectorized(points[:,:2], (heading - 90) * np.pi/180, point_centre)
    points[:,:2] = points[:,:2] - point_centre

    if points.shape[1] >= 3:
        # 2 - heading
        points[:,2] = points[:,2] - heading

    if points.shape[1] >= 4:
        # 3 - velocity_x
        # 4 - velocity_y
        points[:,3:5] = rotate_around_point_highperf_vectorized(points[:,3:5], (heading - 90) * np.pi/180, (0,0))
        pass
    return points

def convert_point_from_radial_to_cartesian(theta, r, point_centre = np.array([0, 0])):
    xy = r * np.array([np.cos(theta), np.sin(theta)]).T + point_centre
    return xy

def sample_points_uniformly_from_the_polygon(polygon, n_points):
    (min_x, min_y) = np.min(polygon, axis=0)
    (max_x, max_y) = np.max(polygon, axis=0)

    points = []
    while len(points) < n_points:
        random_point = [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
        if check_if_point_is_in_polygon_defined_by_corner_points(random_point, polygon):
            points.append(random_point)
    return points

def find_points_uniformly_spread_in_the_polygon(polygon):
    (min_x, min_y) = np.min(polygon, axis=0)
    (max_x, max_y) = np.max(polygon, axis=0)
    x_range = np.arange(min_x, max_x, 1.5)
    y_range = np.arange(min_y, max_y, 1.5)

    #Prune undrivable points
    points = []
    for x in x_range:
        for y in y_range:
            if check_if_point_is_in_polygon_defined_by_corner_points([x, y], polygon):
                points.append(np.array([x, y]))

    if len(points) == 0:
        points = [np.array([np.nan, np.nan])] #
    return points

# def find_points_uniformly_spread_in_the_polygon(polygon, n_points):
#     (min_x, min_y) = np.min(polygon, axis=0)
#     (max_x, max_y) = np.max(polygon, axis=0)
#     x_range = np.arange(min_x, max_x, 1.5)
#     y_range = np.arange(min_y, max_y, 1.5)

#     #Prune undrivable points

#     for x in x_range:
#         for y in y_range:
#             if check_if_point_is_in_polygon_defined_by_corner_points([x, y], polygon):
#                 ax.scatter(x, y, color='red', s=1)


def check_if_point_is_in_polygon_defined_by_corner_points(point, polygon_corners):
    x, y = point
    n = len(polygon_corners)
    inside = False

    p1x, p1y = polygon_corners[0]
    for i in range(n+1):
        p2x, p2y = polygon_corners[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside

def fill_area_between_lines(ax, line1, line2, color):
    x1, y1 = line1
    x2, y2 = line2
    ax.fill_between(x1, y1, y2, color=color, alpha=0.2)   

def point_in_rectangle(M, A, B, C, D):
    # A, B, D need to neighboring
    AM = M - A
    AB = B - A
    AD = D - A
    
    dot_1 = np.dot(AM, AB)
    dot_2 = np.dot(AM, AD)
    if (0 <= dot_1 and dot_1 <= np.dot(AB, AB)):
        if (0 <= dot_2 and dot_2 <= np.dot(AD, AD)):
            return True
    return False

def fill_the_rectangle(A, B, C, D):
    x_min = np.min(np.array([A[0], B[0], C[0], D[0]]))
    x_max = np.max(np.array([A[0], B[0], C[0], D[0]]))
    
    y_min = np.min(np.array([A[1], B[1], C[1], D[1]]))
    y_max = np.max(np.array([A[1], B[1], C[1], D[1]]))
    
    points = []
    
    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            M = np.array([x, y])
            if point_in_rectangle(M, A, B, C, D):
                points.append(M)                
    return points

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) >= (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def point_in_polygon(pt, poly, inf):
    result = False
    for i in range(poly.shape[0]-1):
        if intersect((poly[i,0], poly[i,1]), ( poly[i+1,0], poly[i+1,1]), (pt[0], pt[1]), (inf, pt[1])):
            result = not result
    if intersect((poly[-1,0], poly[-1,1]), (poly[0,0], poly[0,1]), (pt[0], pt[1]), (inf, pt[1])):
        result = not result
    return result

def find_nearest(n,v,v0,vn,res):
    if type(v) is np.ndarray:
        idx = np.int16(np.floor(n*(v-v0+res/2.)/(vn-v0+res)))
    else:
        idx = int(np.floor( n*(v-v0+res/2.)/(vn-v0+res) ))

    return idx

def linefunction(velx,vely,indx,indy,xrange):
    m = (indy-vely)/(indx-velx)
    b = vely-m*velx
    return m*xrange + b

def global_grid(origin,endpoint,res):
    # create a grid of x and y values that have an associated x,y centre
    # coordinate in the global space (42.7 m buffer around the edge points)
    xmin = min(origin[0],endpoint[0]) #-(128)
    xmax = max(origin[0],endpoint[0]) #+res#+(128)+res
    ymin = min(origin[1],endpoint[1])#-(128)
    ymax = max(origin[1],endpoint[1]) #+res #+(128)+res
    x_coords = np.arange(xmin,xmax,res)
    y_coords = np.arange(ymin,ymax,res)
    gridx,gridy = np.meshgrid(x_coords,y_coords)

    return gridx.T,gridy.T
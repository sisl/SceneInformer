import numpy as np

from sceneinformer.utils.geometry_tools import (
    check_if_point_is_in_polygon_defined_by_corner_points,
    convert_point_from_radial_to_cartesian,
    rotate_around_point_highperf_vectorized)

obj_types = {
    'vehicle': 0,
    'pedestrian': 1,
    'cyclist': 2
}

maps_types = {
    'stop_sign': 0,
    'speed_bump': 1,
    'road_edge': 2,
    'crosswalk': 3,
    'road_line': 4,
    'lane': 5
}

traffic_lights_types = {
    'unknown': 0,
    'arrow_stop': 1,
    'arrow_caution': 2,
    'arrow_go': 3,
    'stop': 4,
    'caution': 5,
    'go': 6,
    'flashing_stop': 7,
    'flashing_caution': 8,   
}

MAX_DISTANCE = 50

def get_road(roads):
    all_maps = []
    mapping = []
    for polyline in roads:
        geometry = polyline['points']
        vec_type = polyline['old_type']
        if vec_type == 'driveway':
            continue
        mapping.append(maps_types[vec_type])

        num_points = len(geometry)

        arr = []
        init = True
        for idx, point in enumerate(geometry):

            point = (point['x'], point['y'])
            point = np.array([point[0], point[1]])
            point = np.round(point, 2)

            if vec_type not in ['crosswalk', 'stop_sign', 'speed_bump']:
                if init:
                    init = False
                    arr.append(point)
                    prev_pt = point
                elif idx == num_points - 1:
                    arr.append(point)
                    prev_pt = point
                else:
                    if np.linalg.norm(point - prev_pt) > 1.5:
                        arr.append(point)
                        prev_pt = point
            else:
                arr.append(point)

        arr = np.stack(arr, axis=0)

        assert arr.shape[1] == 2, "Shape of arr is {}".format(arr.shape) #Should be (N, 2)
    
        all_maps.append(arr)

    mapping = np.array(mapping)

    assert mapping.shape[0] == len(all_maps), "Shape of mapping is {}".format(mapping.shape) and "Length of all_maps is {}".format(len(all_maps))

    maps = {
        'points': all_maps,
        'mapping': mapping
    }
    return maps

def get_trajectories(objects):
    trajectories = []
    for obj in objects:
        arr = np.zeros((91,2))
        t_start = 0
        for t in range(91):
            if obj["position"][t]["x"] == -1:
                t_start = t + 1

            arr[t,0] = obj["position"][t]["x"]
            arr[t,1] = obj["position"][t]["y"]

        trajectories.append(tuple([arr, t_start]))

    return trajectories

def normalize(points, heading, point_centre):
    points = rotate_around_point_highperf_vectorized(points, (heading - 90) * np.pi/180, point_centre)
    points = points - point_centre
    return points

def extract_occlusion_from_obj_id_persepective(other_trajectories, dims, t, heading, point_centre, obj_id):
    #The logic before extracts an occluded polygon caused by an agent. It's not pretty :( but it works.
    invalid = np.where(other_trajectories[:,0] == -1)
    other_trajectories[:,:2] = normalize(np.copy(other_trajectories[:,:2]), heading, point_centre) #(x, y, heading)

    dims = dims[:,:2] # (width, length)

    #Prune trajectories that are too far away
    invalid_distance = np.where(np.linalg.norm(other_trajectories[:,:2], axis=1) >= MAX_DISTANCE) 

    # Corners of the vehicles 
    pos_bl = other_trajectories[:,:2] - np.array([[-1/2, -1/2]])*dims
    pos_br = other_trajectories[:,:2] - np.array([[1/2, -1/2]])*dims
    pos_tl = other_trajectories[:,:2] - np.array([[-1/2, 1/2]])*dims
    pos_tr = other_trajectories[:,:2] - np.array([[1/2, 1/2]])*dims

    obj_corners = np.concatenate([pos_bl, pos_br, pos_tr, pos_tl], axis=0) #(N*4, 2)
    obj_centers = np.concatenate(4*[other_trajectories], axis=0)
    obj_corners = rotate_around_point_highperf_vectorized(obj_corners, (heading - obj_centers[:,2]) * np.pi/180, obj_centers[:,:2])

    all_corners = obj_corners.reshape(4,-1,2).transpose(1,0,2)
    all_corners[invalid] = np.inf
    all_corners[invalid_distance] = np.inf
    angles = np.arctan2(all_corners[:,:,1], all_corners[:,:,0]) # (N, 4)
    min_ang = np.min(angles, axis=1) # (N,)
    max_ang = np.max(angles, axis=1) # (N,)
 
    assert angles.shape == (other_trajectories.shape[0], 4)
    assert min_ang.shape == max_ang.shape == (other_trajectories.shape[0],)

    dis_arg = (min_ang < -np.pi/2) * (max_ang > np.pi/2)
    args = np.argwhere(dis_arg)[:,0]

    if args.shape[0] != 0:
        new_angles = angles[args]
        pos_ang = np.copy(new_angles)
        pos_ang[pos_ang < 0] = np.inf
        neg_ang = np.copy(new_angles)
        neg_ang[neg_ang > 0] = -np.inf  
        max_ang[args] = np.min(pos_ang, axis=1)
        min_ang[args] = np.max(neg_ang, axis=1)

    #Radius edge
    points_min = convert_point_from_radial_to_cartesian(min_ang, 60)
    points_max = convert_point_from_radial_to_cartesian(max_ang, 60)

    #all_corners, angles, min_ang, max_ang 
    num_occluding_objects = all_corners.shape[0]
    obj_centers = obj_centers[:num_occluding_objects,:2]

    occlusions = []

    for idx in range(num_occluding_objects):
        corners = all_corners[idx] # (pos_bl -> pos_br -> pos_tr -> pos_tl)

        if np.isinf(corners).any():
            continue

        if np.linalg.norm(other_trajectories[idx,:2]) >= MAX_DISTANCE:
            continue
        
        angle = angles[idx]
        min_ang_obj = min_ang[idx]
        point_min = points_min[idx]
        point_max = points_max[idx]
        max_ang_obj = max_ang[idx]
        polygon_corners = []

        min_arg = np.argwhere(angle==min_ang_obj)[0,0]
        max_arg = np.argwhere(angle==max_ang_obj)[0,0]

        polygon_corners.append(corners[max_arg,:])
        polygon_corners.append(point_max)
        polygon_corners.append(point_min)
        polygon_corners.append(corners[min_arg,:])
        #Add points in between min and max corners (one or two) 

        indices_left = []
        ang_idx = np.copy(min_arg)
        for i in range(4):
            ang_idx = (ang_idx + 1) % 4
            if ang_idx != max_arg and ang_idx != min_arg:
                indices_left.append(ang_idx)
            
        diff = np.abs(max_arg - min_arg)
        if diff == 2:
            val = -np.Inf
            arg_to_be_added = None
            for arg in indices_left:
                val_candidate = np.linalg.norm(corners[arg])
                if val_candidate > val:
                    val = val_candidate
                    arg_to_be_added = arg
            polygon_corners.append(corners[arg_to_be_added,:])

        elif diff == 1 or diff == 3:
            #Two points in between
            if np.linalg.norm(corners[indices_left[0]] - corners[min_arg]) < np.linalg.norm(corners[indices_left[1]] - corners[min_arg]):
                polygon_corners.append(corners[indices_left[0],:])
                polygon_corners.append(corners[indices_left[1],:])
            else:
                polygon_corners.append(corners[indices_left[1],:])
                polygon_corners.append(corners[indices_left[0],:])

        polygon_corners = np.stack(polygon_corners, axis=0)
       
        occlusion_data = {
            'polygon_corners': polygon_corners,
            'centre': obj_centers[idx], 
            'occluding_object_id': idx, 
            'time': t,
            'occluded_object_ids': [],
        }

        occlusions.append(occlusion_data)

    occlusions_from_single_perspective = {
        'occlusions_from_single_perspective': occlusions,
        'object_id_perspective': obj_id
    }

    return occlusions_from_single_perspective


def get_occlusions(objects, t, valid_perspectives):

    all_objects = np.copy(objects['trajectories'][:,t])
    dims = np.copy(objects['mapping']) 

    all_occlusions = []
    
    for obj_id in valid_perspectives:
        point_centre = all_objects[obj_id,:2]
        point_centre = (point_centre[0], point_centre[1])
        heading = all_objects[obj_id,2]

        if point_centre[0] == np.inf or point_centre[0] == -1:
            continue

        other_trajectories = np.copy(all_objects)
        other_trajectories[obj_id] = np.array([-1, -1, -1, -1, -1, -1]) #We are observing from the perspective of this object.

        occlusions_per_agent = extract_occlusion_from_obj_id_persepective(other_trajectories, dims, t, heading, point_centre, obj_id)
        all_occlusions.append(occlusions_per_agent)

    # Extract occluded objects from each occlusion perspective (different agents causing occlusions). 
    for occlusions_from_id_perspective in all_occlusions:

        ego_id = occlusions_from_id_perspective['object_id_perspective']
        trajectories = np.copy(objects['trajectories'][:,t])

        point_centre = trajectories[ego_id,:2]
        point_centre = (point_centre[0], point_centre[1])
        heading = trajectories[ego_id,2]

        trajectories_local_frame = normalize(np.copy(trajectories[:,:2]), heading, point_centre) #(x, y, heading)

        for occlusion in occlusions_from_id_perspective['occlusions_from_single_perspective']:
            occluding_obj_id = occlusion['occluding_object_id']
            polygon_corners = occlusion['polygon_corners']

            for obj_id in range(trajectories_local_frame.shape[0]):
                if obj_id == occluding_obj_id:
                    continue

                pos = trajectories_local_frame[obj_id]

                if check_if_point_is_in_polygon_defined_by_corner_points(pos, polygon_corners):
                    occlusion['occluded_object_ids'].append(obj_id)

    return all_occlusions

def get_objects(objects):
    all_trajectories = []
    mapping = []

    for idx, obj in enumerate(objects):
        obj_traj = []
        obj_type = obj["type"]
        length = obj["length"]
        width = obj["width"]
        mapping.append(np.array([width, length, obj_types[obj_type]]))

        for t in range(91):
            angle = obj["heading"][t]
            point = obj['position'][t]
            vel_x = obj['velocity'][t]['x']
            vel_y = obj['velocity'][t]['y']
            valid_state = obj['valid'][t]
        
            pos = np.array([point['x'], point['y'], angle, vel_x, vel_y, valid_state])
            pos = np.round(pos, 2)
            obj_traj.append(pos)
        obj_traj = np.stack(obj_traj, axis=0) #(91, 3)

        assert obj_traj.shape[0] == 91, "Trajectory length should be 91"
        assert obj_traj.shape[1] == 6, "Trajectory should have 6 dimensions"

        all_trajectories.append(obj_traj)

    all_trajectories = np.stack(all_trajectories, axis=0) #(num_objects, 91, 3)

    assert all_trajectories.shape[0] == len(objects), "Number of objects should be equal to number of trajectories"
    assert all_trajectories.shape[1] == 91, "Trajectory length should be 91"
    assert all_trajectories.shape[2] == 6, "Trajectory should have 6 dimensions"

    mapping = np.stack(mapping, axis=0) #(num_objects, 3)

    assert mapping.shape[0] == len(objects), "Number of objects should be equal to number of trajectories"
    assert mapping.shape[1] == 3, "Mapping should have 3 dimensions"

    objects_to_be_saved = {
        'trajectories': all_trajectories,
        'mapping': mapping
    }

    return objects_to_be_saved
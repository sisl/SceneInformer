import os
import random
from typing import Any, Dict, Iterable, List, Tuple, Union

import h5py
import numpy as np
import torch
from sceneinformer.utils.geometry_tools import (
    find_points_uniformly_spread_in_the_polygon, normalize)
from torch.utils.data import Dataset

MAX_SAMPLED_ANCHORS = 500
# MAX_OBS_AGENTS = 60
# MAX_INF_AGENTS = 60
MAX_POLYGON_CORNERS = 6
MAX_POLYLINES = 500
MAX_POLYLINES_LENGTH = 70
MAX_OBS_AGENT_DIS = 50
MAX_POLY_DIS = 90

POS_X: int = 0
POS_Y: int = 1
HEADING: int = 2
VEL_X: int = 3
VEL_Y: int = 4
VALID: int = 5
DIF_X: int = 6
DIF_Y: int = 7
DIF_HEADING: int = 8
WIDTH: int = 9
LENGTH: int = 10
OBJECT_TYPE: int = 11

def convert(obj: dict) -> dict:
    """
    Recursively converts all the elements of a dictionary to a dictionary of tensors.
    """
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        # check if the dtype is some kind of float
        if np.issubdtype(obj.dtype, np.floating):
            return obj.astype(np.float32)
        else:
            return obj 
    else:
        return obj
    
class VectorizedDatasetHDF5(Dataset):
    def __init__(self, configs: dict = None) -> None:
        super(VectorizedDatasetHDF5, self).__init__()
        self.mode = configs['mode']
        self.path = os.path.join(configs['path'], self.mode)
        self.files = os.listdir(self.path)
        self.t_past = configs['t_past']
        self.t_last_observed = self.t_past - 1 
        self.t_eval = self.t_past - 1
        self.t_future = configs['t_future']
        self.full_obs = configs['full_obs']
        self.occlusion_inf = configs['occlusion_inf']
        self.prob_occupied_occlusion = configs['prob_occupied_occlusion']
        self.dataset_summary = configs['dataset_summary']

        if 'prob_occluding_object' in configs.keys():
            self.prob_occluding_object = configs['prob_occluding_object']
        else:
            self.prob_occluding_object = 1.0

        if 'only_anchor_occluded_objects' in configs.keys():
            self.only_anchor_occluded_objects = configs['only_anchor_occluded_objects'] #Only reasoning about the occlusions
        else:
            self.only_anchor_occluded_objects = False
        
        if 'debug' in configs.keys():
            self.debug = configs['debug']
        else:
            self.debug = False

        if self.mode == 'training':
            self.scene_count = 1000
        elif self.mode == 'validation':
            self.scene_count = 150

        self.load_dataset_summary()

    def load_dataset_summary(self):
        print(f'Loading the dataset from {self.path}.')
        self.scene_samples = []
        for scene in os.listdir(self.path):
            if scene.split('_')[0] == 'dataset' or scene.split('_')[0] == 'sequence':
                continue
            for sample in os.listdir(os.path.join(self.path, scene)):
                self.scene_samples.append((scene, sample))
        print(f'Loaded {len(self.scene_samples)} samples.')

    def __len__(self) -> int: 
        return len(self.scene_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene, sample = self.scene_samples[idx]
        scene_id = int(sample.split('-')[1])
        sample_id = int(sample.split('_')[-1].split('.')[0])
        self.idx = idx

        if np.random.choice([True, False], p=[self.prob_occupied_occlusion, 1 - self.prob_occupied_occlusion]):
            occ = True
            data = np.load(os.path.join(self.path, self.dataset_summary, f'{scene_id}_{sample_id}_dataset_summary_positive_samples_{self.mode}.npy'))       
        else:
            occ = False
            data = np.load(os.path.join(self.path, self.dataset_summary, f'{scene_id}_{sample_id}_dataset_summary_negative_samples_{self.mode}.npy'))            

        num_occlusions = len(data)
        if num_occlusions == 0:
            #check to detect broken samples
            return None

        data_sample_idx = np.random.randint(num_occlusions)
        t, object_id_perspective, occlusion_idx = data[data_sample_idx]
        filename = os.path.join(scene, sample)
        path = os.path.join(self.path, filename)
        sample = self.load_sample(path, t, ego_idx=object_id_perspective, occlusion_idx=occlusion_idx)

        if sample is None:
            #check to detect broken samples
            return None
        processed_sample = self.process_sample(sample, t, object_id_perspective, occlusion_idx) 
        
        if processed_sample is not None:
            processed_sample['id'] = f'{scene_id}_{sample_id}_{occ}_{data_sample_idx}'



        return processed_sample
     
    def load_sample(self, 
                    path: str, 
                    t: int, 
                    ego_idx: int, 
                    occlusion_idx: int = None) -> Dict[str, Any]:
        
        self.path_idx = path
        with h5py.File(path, 'r') as f:
            trajectories = f['/objects/trajectories'][:, t - self.t_past + 1:t + self.t_future + 1, :]
            trajectories_mapping = f['/objects/mapping'][:]
            maps_points = f['/maps/points'][:]
            maps_mapping = f['/maps/mapping'][:]
            perspective_ids = f['/occlusions/object_id_perspective'][t]

            perspective_idx = None
            for _idx, perspective_candidate in enumerate(perspective_ids):
                if perspective_candidate == ego_idx:
                    perspective_idx = _idx
                    break 

            if perspective_idx is None:
                return None


            #Occluding object from the perspective of the ego vehicle.
            occluding_object_id = f['/occlusions/occluding_object_id'][t - self.t_past + 1:t + self.t_future + 1, perspective_idx] #TODO: check timesteps

            #Occluded objects from the perspective of the ego vehicle.
            occluded_object_ids = f['/occlusions/occluded_object_ids'][t - self.t_past + 1:t + self.t_future + 1, perspective_idx]

            polygon_corners = f['/occlusions/polygon_corners'][t - self.t_past + 1:t + self.t_future + 1, perspective_idx] 

            polygon_corners[polygon_corners == 0] = np.nan

            sdc_idx = f['/valid_perspectives'][-1]

            valid_perpectives = f['/valid_perspectives'][:]

            data = {
                'trajectories': trajectories,
                'trajectories_mapping': trajectories_mapping,
                'maps_points': maps_points,
                'maps_mapping': maps_mapping,
                'occluding_object_id': occluding_object_id.astype(np.int32),
                'occluded_object_ids': occluded_object_ids.astype(np.int32),
                'polygon_corners': polygon_corners,
                'sdc_idx': sdc_idx,
                'ego_idx': ego_idx,
                'perspective_idx': perspective_idx,
                'valid_perspectives': valid_perpectives,
                't': t,
            }

        return data
        
    def process_sample(self, 
                       data_sample: Dict[str, Any], 
                       t: int, 
                       ego_idx: int, 
                       occlusion_idx: int = None) -> Dict[str, Any]:
        ego_position = np.copy(data_sample['trajectories'][ego_idx,self.t_last_observed,:2]) 
        ego_heading = np.copy(data_sample['trajectories'][ego_idx,self.t_last_observed,2])

        anchors, polygon_corners, objects_to_be_inferred, occluded_objects, occluding_object_id, all_polygon_corners = self.get_occlusions(data_sample, 
                                                                                                                    occlusion_idx, 
                                                                                                                    ego_position, 
                                                                                                                    ego_heading)
        if self.occlusion_inf:
            if anchors is None:
                return None
        else:
            anchors = np.zeros((0,2))

        num_occluded_anchors = anchors.shape[0]
        trajectories, labels, anchors, org_obs, org_labels, non_nan_obs_anchor_counts, objects_to_be_predicted = self.get_objects(data_sample, 
                                                                                                                                occluded_objects, 
                                                                                                                                objects_to_be_inferred, 
                                                                                                                                t,
                                                                                                                                ego_position, 
                                                                                                                                ego_heading, 
                                                                                                                                anchors, 
                                                                                                                                polygon_corners, 
                                                                                                                                occluding_object_id)
        
        polylines = self.get_maps(data_sample, t, ego_position, ego_heading)
        
        if self.debug:
            num_unoccluded_anchors = np.array([labels.shape[0]]) 
            num_occupied_anchors = len(objects_to_be_inferred) + non_nan_obs_anchor_counts 
            num_free_anchors = num_occluded_anchors - len(objects_to_be_inferred)
            sdc_idx = data_sample['valid_perspectives'][-1]

            sample = {
                'ego_idx': np.array([ego_idx]),
                'occluding_object_id': np.array([occluding_object_id]),
                'anchors': anchors.astype(np.float32),
                'polygon_corners': polygon_corners,
                'observed_trajectories': trajectories.astype(np.float32),
                'labels': labels.astype(np.float32), 
                'polylines': polylines.astype(np.float32),
                'num_unoccluded_anchors': num_unoccluded_anchors,   
                'num_occluded_anchors': np.array([num_occluded_anchors]),
                'data_idx': np.array([self.idx]),
                'sdc_idx': np.array([sdc_idx], dtype=np.int32),
                'org_obs': org_obs,
                'org_labels': org_labels,
                'num_occupied_anchors': np.array([num_occupied_anchors]),
                'num_free_anchors': np.array([num_free_anchors]),
                'all_polygon_corners': all_polygon_corners,
                }
        else:
            sample = {
                'ego_idx': np.array([ego_idx]),
                'occluding_object_id': np.array([occluding_object_id]),
                'num_occluded_anchor': np.array([num_occluded_anchors]),
                'anchors': anchors.astype(np.float32),
                'polygon_corners': polygon_corners,
                'observed_trajectories': trajectories.astype(np.float32),
                'labels': labels.astype(np.float32), 
                'polylines': polylines.astype(np.float32),
                }
            
        return sample

    def get_objects(self, 
                    sample: Dict[str, np.ndarray], 
                    occluded_objects: np.ndarray, 
                    objects_to_be_predicted: np.ndarray, 
                    t: int, point_centre: Tuple[float, float], 
                    heading: float, 
                    anchors: np.ndarray, 
                    polygon_corners: np.ndarray, 
                    occluding_object: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        trajectories = sample['trajectories']
        mapping = sample['trajectories_mapping'] 

        #################### Normalize the trajectories. ####################
        B = trajectories.shape[0]
        time_interval = trajectories.shape[1]
        invalid_trajectories = (trajectories[:,:,0] == -1.0) #-1.0 indicate invalid value.
        trajectories = trajectories.reshape(B*time_interval, -1) 
        point_centre = (point_centre[0], point_centre[1])
        trajectories[:,:5] = normalize(np.copy(trajectories[:,:5]), heading, point_centre) 
        trajectories = trajectories.reshape(B, time_interval, -1)
        trajectories[invalid_trajectories] = np.nan # set invalid values to nan.
    
        #################### Add deltas + dimensions + types to trajectories. ####################
        diff_trajectories = np.zeros_like(trajectories[:,:,:3])
        diff_trajectories[:,:-1] = np.diff(trajectories[:,:,:3], axis=1)
        trajectories = np.concatenate([trajectories, diff_trajectories, mapping[:,np.newaxis,:3].repeat(time_interval,1)], axis=2)
        
        #################### Delete trajectories that are beyond MAX_OBS_AGENT_DIS. ####################
        unobservable = np.argwhere((np.linalg.norm(trajectories[:,:,:2], axis= 2) >= MAX_OBS_AGENT_DIS)) 
        trajectories[unobservable[:,0], unobservable[:,1], :] = np.nan

        #################### Split the trajectories between observations and labels. ####################
        observations = trajectories[:,:self.t_past] # !There is an overlap on purpose between observations and labels!
        labels = (trajectories[:,self.t_past - 1:]).copy()

        assert observations.shape[1] == self.t_past, f'Observations shape is {observations.shape[1]} but should be {self.t_past}.'
        assert labels.shape[1] == self.t_future + 1, f'Labels shape is {labels.shape[1]} but should be {self.t_future + 1}.' 

        if self.debug:
            org_observations = observations.copy()
            org_labels = labels.copy()
        else:
            org_observations = None
            org_labels = None

        #################### Observations processing. Switch between "full-visibilty" setting and "limited-visibility" setting. ####################
        if not self.full_obs: 
            for t in range(self.t_past): 
                occlusions_per_t = occluded_objects[t]
                for _occlusion in occlusions_per_t:
                    _occluded_objects = _occlusion["occluded"]

                    for obj_id in _occluded_objects:
                        # print(f'Pruning object:{obj_id} at time:{t}.')
                        observations[int(obj_id),t, :] = np.nan
                        
                        if t == self.t_past - 1 and int(obj_id) not in objects_to_be_predicted: 
                            labels[int(obj_id),0,:] = np.nan 
                        elif t == self.t_past - 1 and int(obj_id) in objects_to_be_predicted:
                            pass
        else:
            if self.occlusion_inf: #We only delete observations that are within the occlusion of interest.
                for t in range(self.t_past): 
                    occlusions_per_t = occluded_objects[t]
                    for _occlusion in occlusions_per_t:
                        _occluding_object_id = _occlusion["occluding"]
                        _occluded_objects = _occlusion["occluded"]
                        
                        if _occluding_object_id == occluding_object:
                            for obj_id in _occluded_objects:
                                observations[int(obj_id),t, :] = np.nan
                                pass
            else:
                pass

        #################### Add anchors for the observed objects. For observed objects, anchors are the last observed position. ####################
        occupied_anchors = np.arange(labels.shape[0]) + anchors.shape[0]
        occupied_anchors = occupied_anchors[:,np.newaxis,np.newaxis].repeat(labels.shape[1], 1) 

        # # If we don't see the object at t_eval and it is not in the occluded objects list, delete it.
        obs_anchors = np.copy(observations[:, -1, :2]) # -1 is t_eval.
        nans_anchors = np.isnan(obs_anchors).any(1)
        for i in range(nans_anchors.shape[0]):
            if nans_anchors[i]:  #np.isnan(observations[i]).all():
                if i not in objects_to_be_predicted:
                    labels[i] = np.nan 
                continue
        non_nan_anchor_count = np.sum(~np.isnan(obs_anchors[:,0]))

        if self.only_anchor_occluded_objects:
            obs_anchors[:,:] = np.nan
  
        #################### Prepare occluded anchors. ####################

        time_interval = labels.shape[1]
        labels = np.concatenate([labels, occupied_anchors], axis=2) # Add occupied anchors to the labels.

        # Filter out the objects to be predicted that are beyond the obserable region.
        if len(objects_to_be_predicted) != 0:
            modified_objects_to_be_predicted = []
            for _id in objects_to_be_predicted:
                if not np.isnan(labels[_id, 0, :2]).any():
                    # Objects is not observable at t_eval.
                    modified_objects_to_be_predicted.append(_id)
                else:
                    pass

            objects_to_be_predicted = np.array(modified_objects_to_be_predicted)
        
        if len(objects_to_be_predicted) != 0:
            objects_to_be_predicted = np.array(objects_to_be_predicted)
            occluded_labels = labels[objects_to_be_predicted]
            time_interval = labels.shape[1]
            occ_anchor_pos = occluded_labels[:,0,:2] 
            nans = np.isnan(occ_anchor_pos).any(1) 

            # Check that there are no nans.
            if nans.any():
                return None, None, None, None, None, None

            occ_anchor_pos = occ_anchor_pos[:,np.newaxis,:]
            anchors = anchors[np.newaxis,:,:]
            dis = np.linalg.norm(anchors - occ_anchor_pos, axis=2)

            if np.isnan(dis).any():
                return None, None, None, None, None, None

            occupied_anchors = (np.nanargmin(dis, axis=1)[:,np.newaxis]) #(N,1,1)
            occupied_anchors = occupied_anchors.repeat(time_interval,1) #(N,T,1)
            anchors = anchors[0,:,:]
            illegal_anchors = (labels[objects_to_be_predicted, 0, -1]).astype('int32') 
            labels[objects_to_be_predicted, :,-1] = occupied_anchors
            all_anchors = np.concatenate([anchors, obs_anchors], axis=0)
            all_anchors[illegal_anchors] = np.nan
        else:
            all_anchors = np.concatenate([anchors, obs_anchors], axis=0)

        return observations, labels, all_anchors, org_observations, org_labels, non_nan_anchor_count, objects_to_be_predicted

    def get_occlusions(self, 
                       data_sample: dict, 
                       occlusion_idx: int, 
                       ego_position: np.ndarray, 
                       ego_heading: float) -> Tuple[Union[None, np.ndarray], np.ndarray, np.ndarray, List[List[dict]], int]:

        if not self.full_obs and self.prob_occluding_object >= 0.0:
            considered_occluding_object_ids = []
            all_occluding_object_ids = []
            for t in range(self.t_past + self.t_future):
                for idx, occluding_object_id in enumerate(data_sample['occluding_object_id'][t]):
                    if occluding_object_id == -1:
                        continue
                    if idx != occlusion_idx and occluding_object_id not in considered_occluding_object_ids and np.random.rand() < self.prob_occluding_object:
                        all_occluding_object_ids.append(occluding_object_id)
                    if idx != occlusion_idx and occluding_object_id not in considered_occluding_object_ids:
                        considered_occluding_object_ids.append(occluding_object_id)

            if self.occlusion_inf:
                sampled_occluding_object_id = data_sample['occluding_object_id'][self.t_last_observed, occlusion_idx]

                if sampled_occluding_object_id not in all_occluding_object_ids:
                    all_occluding_object_ids.append(sampled_occluding_object_id)
                    considered_occluding_object_ids.append(sampled_occluding_object_id)

        elif self.full_obs and self.occlusion_inf:
            all_occluding_object_ids = [data_sample['occluding_object_id'][self.t_last_observed, occlusion_idx]]
        else:
            all_occluding_object_ids= None

        if self.occlusion_inf:
            assert all_occluding_object_ids is not None, f'Error 1. All occluding object ids are None:{self.path_idx}. Idx:{self.idx}.\n'

        occluded_objects = []
        tmp_polygon_corners = []
        for t in range(self.t_past + self.t_future):
            occluded_objects_per_t = []
            for idx, occluding_object_id in enumerate(data_sample['occluding_object_id'][t]):
                if occluding_object_id == -1:
                    continue

                if all_occluding_object_ids is None:
                    continue

                if occluding_object_id not in all_occluding_object_ids:
                    continue

                if t == self.t_last_observed:
                    tmp_polygon_corners.append(data_sample['polygon_corners'][t, idx])  


                occluded = data_sample['occluded_object_ids'][t, idx]
                occluded_objects_per_t.append({"occluding": occluding_object_id, "occluded": occluded[occluded != -1]})
            occluded_objects.append(occluded_objects_per_t)
       
        # Get the anchors for the occlusion
        polygon_corners = data_sample['polygon_corners'][self.t_last_observed, occlusion_idx]
        
        if len(tmp_polygon_corners) == 0:
            all_polygon_corners = polygon_corners[None] 
        else:
            all_polygon_corners = np.stack(tmp_polygon_corners, 0) 

        if not self.occlusion_inf:
            return np.zeros((0,2)), np.zeros((0,2)), [], occluded_objects, [], all_polygon_corners #np.zeros((0,0,2))

        polygon_corners = polygon_corners[~np.isnan(polygon_corners).all(axis=1)]
        anchors, polygon_corners = self.get_anchors(polygon_corners, ego_position, ego_heading)

        if anchors is None:
            return None, None, None, None, None, None

        # Get the objects ids we are predicting for the labels.
        occluding_object_id = data_sample['occluding_object_id'][self.t_last_observed, occlusion_idx] 

        objects_to_be_inferred = data_sample['occluded_object_ids'][self.t_last_observed, occlusion_idx]
        objects_to_be_inferred = objects_to_be_inferred[objects_to_be_inferred != -1] #Some of them might be beyond the observable region.

        polygon_corners = self.delete_and_pad_or_sample(polygon_corners[:,None,:], MAX_POLYGON_CORNERS)
        polygon_corners = polygon_corners[:,0,:]

        return anchors, polygon_corners, objects_to_be_inferred, occluded_objects, occluding_object_id, all_polygon_corners
    
    def get_anchors(self, polygon_corners, point_centre, heading):        
        anchors = find_points_uniformly_spread_in_the_polygon(polygon_corners)
        if len(anchors) > MAX_SAMPLED_ANCHORS:
            anchors = random.sample(anchors, MAX_SAMPLED_ANCHORS) #(anchors, MAX_SAMPLED_ANCHORS, replace=False)
        anchors = np.stack(anchors, axis=0)

        if np.isnan(anchors).all():
            anchors = None
 
        return anchors, polygon_corners

    def get_maps(self, maps: Dict[str, np.ndarray], t: int, point_centre: Tuple[float, float], heading: float) -> np.ndarray:
        polylines = maps['maps_points'] 
        Nm = len(polylines)
        invalid = (polylines == -1)
        point_centre = (point_centre[0], point_centre[1])
        polylines = polylines.reshape(-1, 2)
        polylines = normalize(polylines, heading, point_centre)
        polylines = polylines.reshape(Nm, -1, 2)
        polylines[invalid] = -1
        diff_polylines = np.zeros_like(polylines)
        diff_polylines[:,:-1] = np.diff(polylines, axis=1) 
        polylines[invalid] = np.nan
        polylines = np.concatenate([polylines, diff_polylines], axis=2)
        invalid = np.argwhere(np.linalg.norm(polylines[:,:,:2], axis=2) > MAX_POLY_DIS)
        polylines[invalid[:,0], invalid[:,1]] = np.nan

        if np.isnan(polylines).all():
            polylines = None
            return None

        map_types = (maps['maps_mapping']).astype('int32')
        map_types = np.eye(6)[map_types]
        num_points = polylines.shape[1]
        polylines = np.concatenate([polylines, map_types[:,np.newaxis].repeat(num_points,1)], axis=2)
        max_polylines = min(MAX_POLYLINES, polylines.shape[0])
        polylines = self.delete_and_pad_or_sample(polylines, max_polylines)

        return polylines

    def stack_padding(self, it: Iterable[np.ndarray], max_size: int) -> np.ndarray:
        def resize(mat, size):
            adjust_size = size - mat.shape[0]
            if adjust_size > 0:
                #pad with -1
                mat = np.pad(mat, ((0,adjust_size), (0,0)), constant_values=((0,-1), (0,0)))
            else:
                #sample
                if adjust_size < 0:
                    mat = self.delete_and_pad_or_sample(mat[:,None,:], size)
                    mat = mat[:,0,:]                    
            return mat

        mat = np.stack( [resize(row, max_size) for row in it], axis=0 )

        return mat
    
    def stack_padding_polylines(self, it: Iterable[np.ndarray], max_size: int) -> np.ndarray:
        def resize(mat: np.ndarray, size: int) -> np.ndarray:
            adjust_size = size - mat.shape[0]
            if adjust_size > 0:
                mat = np.pad(mat, ((0,adjust_size), (0,0)), constant_values=((0,-1), (0,0)))
            else:
                if adjust_size < 0:
                    indices = np.sort(np.random.choice(mat.shape[0], size, replace=False))
                    mat = mat[indices]
                
            return mat
        mat = np.stack( [resize(row, max_size) for row in it], axis=0 )
        return mat
    
    def delete_and_pad_or_sample(self, tensor: np.ndarray, MAX_VAL: int) -> np.ndarray:
        tensor = np.delete(tensor, np.argwhere(np.isnan(tensor).all((1,2))), axis=0) 
        # Pad or sample top-k observations and labels
        if tensor.shape[0] < MAX_VAL:
            tensor = np.pad(tensor, ((0,MAX_VAL - tensor.shape[0]),(0,0), (0,0)), 'constant', constant_values=np.nan)
        elif tensor.shape[0] > MAX_VAL:
            min_tensor = np.min(tensor, axis=1)[:,:2]
            dis = np.linalg.norm(min_tensor, axis=1) #(N,T,2)
            tensor = tensor[np.argpartition(dis, MAX_VAL)][:MAX_VAL]
        return tensor
    
    def get_scene_info(self, scene: int, sample: int, t: int) -> None:
        """
        Prints information about the scene.

        Args:
            scene (int): Scene ID.
            sample (int): Sample ID.
            t (int): Time step.
        """
        filename =  f'{str(scene).zfill(5)}/{self.mode}.tfrecord-{str(scene).zfill(5)}-of-{self.scene_count:05}_{sample}.h5' 
        path = os.path.join(self.path, filename)
        print(f'############## Sample info ##############')
        with h5py.File(path, 'r') as f:
            valid_perspectives = f['/valid_perspectives'][:]
            occluding_object_id = f['/occlusions/occluding_object_id'][t, :]
            for idx in range(occluding_object_id.shape[0]):
                print(f'P: {valid_perspectives[idx]}. Occluding object: {occluding_object_id[idx]}')
    
    def get_scene_frame(self, 
                    scene: int, 
                    sample: int, 
                    t: int, 
                    object_id_perspective: int, 
                    occlusion_idx: int = None) -> Dict[str, Any]:
        """
        Returns the processed sample for the given scene, sample, and time step.

        Args:
            scene (int): Scene ID.
            sample (int): Sample ID.
            t (int): Time step.
            object_id_perspective (int): Object ID.
            occluding_object_id (int, optional): Occluding object ID. Defaults to None.

        Returns:
            Dict[str, Any]: Processed sample.
        """
        filename = f'{str(scene).zfill(5)}/{self.mode}.tfrecord-{str(scene).zfill(5)}-of-{self.scene_count:05}_{sample}.h5'    
        path = os.path.join(self.path, filename)
        self.idx = -1
        sample = self.load_sample(path, t, ego_idx=object_id_perspective, occlusion_idx=occlusion_idx)
        output = self.process_sample(sample, t, object_id_perspective, occlusion_idx)

        output['id'] = f'{scene}_{sample}_{-1}_{-1}'

        # Convert to torch tensors and add batch dimension.
        for key in output.keys():
            if key != 'id':
                output[key] = torch.from_numpy(output[key]).unsqueeze(0)
        return output
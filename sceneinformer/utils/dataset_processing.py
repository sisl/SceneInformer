import pickle

import h5py
import numpy as np


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save(h5file: h5py.File, path: str, mat: np.ndarray, compression: str = "gzip", chunks: bool = None) -> None:
    """
    Saves a numpy array to an HDF5 file at the specified path.

    Args:
        h5file (h5py.File): The HDF5 file to save the data to.
        path (str): The path to the dataset in the HDF5 file.
        mat (np.ndarray): The numpy array to save.
        compression (str, optional): The compression algorithm to use. Defaults to "gzip".
        chunks (bool, optional): Whether to use chunking. Defaults to None.
    """
    if chunks is None and mat.shape[0] > 0:
        chunks = mat.shape
    chunks = True
    h5file.create_dataset(
        path,
        data=mat,
        compression=compression,
        compression_opts=9,
        chunks=chunks
    )

def recursively_save_dict_contents_to_group(h5file: h5py.File, path: str, dic: dict) -> None:
    """
    Recursively saves the contents of a dictionary to an HDF5 file at the specified path.

    Args:
        h5file (h5py.File): The HDF5 file to save the data to.
        path (str): The path to the dataset in the HDF5 file.
        dic (dict): The dictionary to save.
    """
    if len(path) != 0:
        check = path.split('/')[-2]
        if check == 'valid_perspectives':
            dic = np.array(dic)
    
    if isinstance(dic, (np.ndarray, np.int64, np.float64, str, bytes)):
        # If it's numpy array, save it as a dataset
        save(h5file, path, dic)
        return
    
    if isinstance(dic, (list, tuple)):
        if all(isinstance(x, (int, float)) for x in dic):
            # If it's a list or tuple of ints or floats, save it as a dataset
            mat = np.array(dic)
            save(h5file, path, mat)
            return
    
    if isinstance(dic, dict):
        for key, item in dic.items():
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                save(h5file, path + '/' + key, item)
            elif isinstance(item, (dict, list, tuple)):
                recursively_save_dict_contents_to_group(h5file, path + "/" + key, item)
            elif isinstance(item, (int, float)):

                mat = np.array([item])
                save(h5file, path + '/' + key, mat)
            else:
                raise ValueError(f'Cannot save key:{key} with type {type(item)}')

    elif isinstance(dic, (list, tuple)):
        mat = np.array([len(dic)])
        h5file.create_dataset(path + '/length', data=mat, compression="gzip", chunks=mat.shape)
        for i, item in enumerate(dic):
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                save(h5file, path + '/' + str(i), item)
            elif isinstance(item, (dict, list, tuple)):
                recursively_save_dict_contents_to_group(h5file, path + "/" + str(i), item)            
            else:
                raise ValueError(f'Cannot save elem:{i} in path:{path} with type {type(item)}')

    else:
        raise ValueError('Expected dict, list or tuple, got %s type'%type(dic))

def pad_and_stack(arrays: list, d: int = 1, constant: int = 0) -> np.ndarray:
    """
    Pads and stacks a list of arrays.

    Args:
        arrays (list): A list of arrays to pad and stack.
        d (int, optional): The dimensionality of the output array. Defaults to 1.
        constant (int, optional): The constant value to use for padding. Defaults to 0.

    Returns:
        np.ndarray: The padded and stacked array.
    """
    if len(arrays) == 0:
        return np.zeros(tuple([0] * d))
    
    arrays = [np.array(arr) if type(arr) is list else arr for arr in arrays]

    # Add missing dimensions to the beginning of the arrays
    arrays = [arr[np.newaxis] * (d - arr.ndim + 1) if arr.ndim < d else arr for arr in arrays]

    # Find the maximum lengths for each dimension
    max_lens = [max(arr.shape[i] if arr.ndim > i else 0 for arr in arrays) for i in range(max([arr.ndim for arr in arrays]))]

    # Pad arrays
    padded_arrays = []
    for arr in arrays:
        pad_width = [(0, max_lens[i] - arr.shape[i]) for i in range(arr.ndim)]
        arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=constant)
        padded_arrays.append(arr)

    # Stack arrays
    stacked_array = np.stack(padded_arrays)

    assert stacked_array.ndim == d + 1, f'Arrays:{max_lens}. Stacked: {stacked_array.shape}'

    return stacked_array

def flattened_data_structure(data: dict) -> dict:
    """
    A flattened data structure containing the data for all scenes. It's a fix to the previous data format. Ideally, in the future we should change the data format to this one.

    Args:
        data (dict): The data for a single scene.
    """

    data['maps']['points'] = pad_and_stack(data['maps']['points'], d=2, constant=-1) 
    object_id_perspective_list_per_t = []
    occluded_object_ids_list_per_t = []
    occluding_object_id_list_per_t = []
    polygon_corners_list_per_t = []
    for t in range(91):
        object_id_perspective_list = []
        occluded_object_ids_list = []
        occluding_object_id_list = []
        polygon_corners_list = []
        for idx, occ_perspective in enumerate(data['occlusions'][t]):
            object_id_perspective = occ_perspective['object_id_perspective']
            occluded_list, occluding_list, polygon_list = [], [], []
            for occ in occ_perspective['occlusions_from_single_perspective']:
                occluding = occ['occluding_object_id']
                occluded = occ['occluded_object_ids']
                poly = occ['polygon_corners']
                occluded_list.append(occluded)
                occluding_list.append(occluding)
                polygon_list.append(poly)

            occluded_tensor = pad_and_stack(occluded_list, d=1, constant=-1) #pad_and_stack_1d(occluded_list, constant=-1)
            occluding_tensor = np.array(occluding_list)
            polygon_tensor = pad_and_stack(polygon_list, d=2, constant=0) #pad_and_stack_2d(polygon_list, constant=0)

            object_id_perspective_list.append(object_id_perspective)
            occluded_object_ids_list.append(occluded_tensor)
            occluding_object_id_list.append(occluding_tensor)
            polygon_corners_list.append(polygon_tensor)
        
        object_id_perspective_tensor = np.array(object_id_perspective_list)
        occluded_object_ids_tensor = pad_and_stack(occluded_object_ids_list, d=2, constant=-1)
        occluding_object_id_tensor = pad_and_stack(occluding_object_id_list, d=1, constant=-1)  
        polygon_corners_tensor = pad_and_stack(polygon_corners_list, d=3, constant=0) 

        object_id_perspective_list_per_t.append(object_id_perspective_tensor)
        occluded_object_ids_list_per_t.append(occluded_object_ids_tensor)
        occluding_object_id_list_per_t.append(occluding_object_id_tensor)
        polygon_corners_list_per_t.append(polygon_corners_tensor)

    object_id_perspective_t_tensor = pad_and_stack(object_id_perspective_list_per_t, d=1, constant=-1) # Fine to have -1                   
    occluded_object_ids_t_tensor = pad_and_stack(occluded_object_ids_list_per_t, d=3, constant=-1) # Fine to have --1
    occluding_object_id_t_tensor = pad_and_stack(occluding_object_id_list_per_t, d=2, constant=-1) # Fine to have -1
    polygon_corners_t_tensor = pad_and_stack(polygon_corners_list_per_t, d=4, constant=0) #Fine to have 0

    data['occlusions'] = {
        'object_id_perspective': object_id_perspective_t_tensor,
        'occluded_object_ids': occluded_object_ids_t_tensor,
        'occluding_object_id': occluding_object_id_t_tensor,
        'polygon_corners': polygon_corners_t_tensor
    }
    return data
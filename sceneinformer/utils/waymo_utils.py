import enum
import math
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional


import tensorflow as tf
from waymo_open_dataset.protos import map_pb2, scenario_pb2

ERR_VAL = -1

_WAYMO_OBJECT_STR = {
    scenario_pb2.Track.TYPE_UNSET: "unset",
    scenario_pb2.Track.TYPE_VEHICLE: "vehicle",
    scenario_pb2.Track.TYPE_PEDESTRIAN: "pedestrian",
    scenario_pb2.Track.TYPE_CYCLIST: "cyclist",
    scenario_pb2.Track.TYPE_OTHER: "other",
}

_WAYMO_ROAD_STR = {
    map_pb2.TrafficSignalLaneState.LANE_STATE_UNKNOWN: "unknown",
    map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_STOP: "arrow_stop",
    map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_CAUTION: "arrow_caution",
    map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_GO: "arrow_go",
    map_pb2.TrafficSignalLaneState.LANE_STATE_STOP: "stop",
    map_pb2.TrafficSignalLaneState.LANE_STATE_CAUTION: "caution",
    map_pb2.TrafficSignalLaneState.LANE_STATE_GO: "go",
    map_pb2.TrafficSignalLaneState.LANE_STATE_FLASHING_STOP: "flashing_stop",
    map_pb2.TrafficSignalLaneState.LANE_STATE_FLASHING_CAUTION:
    "flashing_caution",
}

# Constants defined for map drawing parameters.
class FeatureType(enum.Enum):
  """Definintions for map feature types."""
  UNKNOWN_FEATURE = 0
  FREEWAY_LANE = 1
  SURFACE_STREET_LANE = 2
  BIKE_LANE = 3
  BROKEN_SINGLE_WHITE_BOUNDARY = 6
  SOLID_SINGLE_WHITE_BOUNDARY = 7
  SOLID_DOUBLE_WHITE_BOUNDARY = 8
  BROKEN_SINGLE_YELLOW_BOUNDARY = 9
  BROKEN_DOUBLE_YELLOW_BOUNDARY = 10
  SOLID_SINGLE_YELLOW_BOUNDARY = 11
  SOLID_DOUBLE_YELLOW_BOUNDARY = 12
  PASSING_DOUBLE_YELLOW_BOUNDARY = 13
  ROAD_EDGE_BOUNDARY = 15
  ROAD_EDGE_MEDIAN = 16
  STOP_SIGN = 17
  CROSSWALK = 18
  SPEED_BUMP = 19
  DRIVEWAY = 20

lane_types = {
      map_pb2.LaneCenter.TYPE_UNDEFINED: FeatureType.UNKNOWN_FEATURE,
      map_pb2.LaneCenter.TYPE_FREEWAY: FeatureType.FREEWAY_LANE,
      map_pb2.LaneCenter.TYPE_SURFACE_STREET: FeatureType.SURFACE_STREET_LANE,
      map_pb2.LaneCenter.TYPE_BIKE_LANE: FeatureType.BIKE_LANE,
  }
road_line_types = {
      map_pb2.RoadLine.TYPE_UNKNOWN: (
          FeatureType.UNKNOWN_FEATURE
      ),
      map_pb2.RoadLine.TYPE_BROKEN_SINGLE_WHITE: (
          FeatureType.BROKEN_SINGLE_WHITE_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_SOLID_SINGLE_WHITE: (
          FeatureType.SOLID_SINGLE_WHITE_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_SOLID_DOUBLE_WHITE: (
          FeatureType.SOLID_DOUBLE_WHITE_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_BROKEN_SINGLE_YELLOW: (
          FeatureType.BROKEN_SINGLE_YELLOW_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: (
          FeatureType.BROKEN_DOUBLE_YELLOW_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_SOLID_SINGLE_YELLOW: (
          FeatureType.SOLID_SINGLE_YELLOW_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_PASSING_DOUBLE_YELLOW: (
          FeatureType.PASSING_DOUBLE_YELLOW_BOUNDARY
      ),
  }
road_edge_types = {
      map_pb2.RoadEdge.TYPE_UNKNOWN: FeatureType.UNKNOWN_FEATURE,
      map_pb2.RoadEdge.TYPE_ROAD_EDGE_BOUNDARY: FeatureType.ROAD_EDGE_BOUNDARY,
      map_pb2.RoadEdge.TYPE_ROAD_EDGE_MEDIAN: FeatureType.ROAD_EDGE_MEDIAN,
  }

def _parse_object_state(
        states: scenario_pb2.ObjectState,
        final_state: scenario_pb2.ObjectState) -> Dict[str, Any]:
    """Construct a dict representing the trajectory and goals of an object.
    Args:
        states (scenario_pb2.ObjectState): Protobuf of object state
        final_state (scenario_pb2.ObjectState): Protobuf of last valid object state.
    Returns
    -------
        Dict[str, Any]: Dict representing an object.
    """
    return {
        "position": [{
            "x": state.center_x,
            "y": state.center_y
        } if state.valid else {
            "x": ERR_VAL,
            "y": ERR_VAL
        } for state in states],
        "width":
        final_state.width,
        "length":
        final_state.length,
        "heading": [
            math.degrees(state.heading) if state.valid else ERR_VAL
            for state in states
        ],  # Use rad here?
        "velocity": [{
            "x": state.velocity_x,
            "y": state.velocity_y
        } if state.valid else {
            "x": ERR_VAL,
            "y": ERR_VAL
        } for state in states],
        "valid": [state.valid for state in states],
    }


def _init_tl_object(track):
    """Construct a dict representing the traffic light states."""
    returned_dict = {}
    for lane_state in track.lane_states:
        returned_dict[lane_state.lane] = {
            'state': _WAYMO_ROAD_STR[lane_state.state],
            'x': lane_state.stop_point.x,
            'y': lane_state.stop_point.y
        }
    return returned_dict


def _init_object(track: scenario_pb2.Track) -> Optional[Dict[str, Any]]:
    """Construct a dict representing the state of the object (vehicle, cyclist, pedestrian).
    Args:
        track (scenario_pb2.Track): protobuf representing the scenario
    Returns
    -------
        Optional[Dict[str, Any]]: dict representing the trajectory and velocity of an object.
    """
    final_valid_index = 0
    for i, state in enumerate(track.states):
        if state.valid:
            final_valid_index = i

    obj = _parse_object_state(track.states, track.states[final_valid_index])
    obj["type"] = _WAYMO_OBJECT_STR[track.object_type]
    obj["id"] = int(track.id)

    return obj

def add_points(
      feature_id: int,
      points: List[map_pb2.MapPoint],
      feature_type: FeatureType,
      is_polygon=False,
) -> Optional[Dict[str, Any]]:
    if feature_type is None:
      return
    
    sample = {
       "id": feature_id,
       "type": feature_type,
       "points": [],
       "is_polygon": is_polygon,
    }
    for point in points:
       sample["points"].append({"x": point.x, "y": point.y})

    if is_polygon:
       sample["points"].append({"x": points[0].x, "y": points[0].y})

    return sample

def _init_road(feature: map_pb2.MapFeature) -> Optional[Dict[str, Any]]:
    """Convert an element of the map protobuf to a dict representing its coordinates and type."""
    feature_old_type = feature.WhichOneof("feature_data")
    
    if feature_old_type is None:
        raise ValueError('feature_old_type is None')

    if feature.HasField('lane'):
       sample = add_points(
          feature.id,
          list(feature.lane.polyline),
          lane_types.get(feature.lane.type),
      )
    elif feature.HasField('road_line'):
      feature_type = road_line_types.get(feature.road_line.type)
      sample = add_points(
          feature.id, list(feature.road_line.polyline), feature_type
      )
    elif feature.HasField('road_edge'):
      feature_type = road_edge_types.get(feature.road_edge.type)
      sample = add_points(
          feature.id, list(feature.road_edge.polyline), feature_type
      )
    elif feature.HasField('stop_sign'):
      sample = add_points(
          feature.id,
          [feature.stop_sign.position],
          FeatureType.STOP_SIGN,
      )
    elif feature.HasField('crosswalk'):
      sample = add_points(
          feature.id,
          list(feature.crosswalk.polygon),
          FeatureType.CROSSWALK,
          True,
      )
    elif feature.HasField('speed_bump'):
      sample = add_points(
          feature.id,
          list(feature.speed_bump.polygon),
          FeatureType.SPEED_BUMP,
          True,
      )
    elif feature.HasField('driveway'):
      sample = add_points(
          feature.id,
          list(feature.driveway.polygon),
          FeatureType.DRIVEWAY,
          True,
      )
    else:
        if feature_old_type is None:
            sample = None
        else:
            print(f'Sample is :{feature}')

    if sample is not None:
       sample["old_type"] = feature_old_type

    return sample  

def load_protobuf(protobuf_path: str) -> Iterator[scenario_pb2.Scenario]:
    """Yield the sharded protobufs from the TFRecord."""
    dataset = tf.data.TFRecordDataset(protobuf_path, compression_type="")
    for data in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(data.numpy()))
        yield scenario


def waymo_to_scenario(scenario_path: str,
                      protobuf: scenario_pb2.Scenario,
                      no_tl: bool = False,
                      include_roads = True) -> None:
    """Dump a JSON File containing the protobuf parsed into the right format.
    Args
    ----
        scenario_path (str): path to dump the json file
        protobuf (scenario_pb2.Scenario): the protobuf we are converting
        no_tl (bool, optional): If true, environments with traffic lights are not dumped.
    """

    # Construct the traffic light states
    tl_dict = defaultdict(lambda: {
        'state': [],
        'x': [],
        'y': [],
        'time_index': []
    })
    tl_dict = {}
    all_keys = ['state', 'x', 'y']
    i = 0

    tracks_to_predict = []
    for track in protobuf.tracks_to_predict:
        tracks_to_predict.append(track.track_index)

    for dynamic_map_state in protobuf.dynamic_map_states:

        traffic_light_dict = _init_tl_object(dynamic_map_state)
        if len(traffic_light_dict) > 0:
            for id, value in traffic_light_dict.items():
                if id not in tl_dict.keys():
                    tl_dict[id] = {
                        'state': [],
                        'x': [],
                        'y': [],
                        'time_index': []
                    }
                for state_key in all_keys:
                    tl_dict[id][state_key].append(value[state_key])
                tl_dict[id]['time_index'].append(i)
            i += 1

    objects = []

    sdc_track_index = protobuf.sdc_track_index
    for idx, track in enumerate(protobuf.tracks):
        obj = _init_object(track)
        # if obj is not None and idx != sdc_track_index:
        #     objects.append(obj)
        # else:
        #     sdc_object = obj
        if obj is not None:
            objects.append(obj)

    # Construct the map states
    if include_roads:
        roads = []
        for map_feature in protobuf.map_features:
            road = _init_road(map_feature)
            if road is not None:
                roads.append(road)
    else:
        roads = None

    scenario = {
        "scenario_id": protobuf.scenario_id,
        "sdc_track_index": sdc_track_index,
        "tracks_to_predict": tracks_to_predict,
        "objects": objects,
        "roads": roads,
        "tl_states": tl_dict
    }
    
    return scenario

def load_protobuf(protobuf_path: str) -> Iterator[scenario_pb2.Scenario]:
    """Yield the sharded protobufs from the TFRecord."""
    dataset = tf.data.TFRecordDataset(protobuf_path, compression_type="")
    for data in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(data.numpy()))
        yield scenario

def return_scenario_list(path: str):
    with tf.device('/CPU:0'):
        scenario_list = []
        for idx, data in enumerate(load_protobuf(path)):
            scenario = waymo_to_scenario(path, data)
            scenario_list.append(scenario)

        return scenario_list
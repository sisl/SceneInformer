import os
import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).parents[2]))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib.path import Path
from sceneinformer.utils.geometry_tools import rotate_around_point_highperf
from scipy.interpolate import griddata

maps_num_types = {
    0:'stop_sign',
    1:'speed_bump',
    2:'road_edge',
    3:'crosswalk',
    4:'road_line',
    5:'lane'
}

colors_num = {
    0: ("H", "gray", 100),
    1: (".", "gray", 2),
    2: (",", "gray", 2),
    3: (".", "gray", 2),
    4: (",", "gray", 1.2),
    5: (",", "gray", 1.2)
}

agents_to_color = {
    0: 'black',
    1: 'orange',
    2: 'green',
}

POS_X = 0
POS_Y = 1
HEADING = 2
VEL_X = 3
VEL_Y = 4
VALID = 5
DIF_X = 6
DIF_Y = 7
DIF_HEADING = 8
WIDTH = 9
LENGTH = 10
OBJECT_TYPE = 11

def generate_visualization(sample, output=None, 
                           vis_path='None', name='None',
                             handpicked_anchors=[], plot_future=False, 
                             visualize_ids=True, just_polygon=False, 
                             just_anchors=False, visibility_only=False, 
                             save=True, 
                             fig=None, ax=None, 
                             xlim=None, ylim=None):
    
    B = sample['observed_trajectories'].shape[0]
    fcn = torch.nn.Softmax(dim=-1)

    for b in range(B):
        single_batch = {}
        for key in sample.keys():
            if key == 'id':
                single_batch[key] = sample[key][b]
                continue
            single_batch[key] = (sample[key][b]).cpu().detach().numpy()
        if output is not None:    
            single_output = {}
            for key in output.keys():
                if key == 'logits_occ':
                    logits_occ = output[key]
 
                    probs_occ = fcn(logits_occ)
                    probs_occ = torch.clamp(probs_occ, min=0, max=1.0).cpu().detach().numpy()
                if key == 'logits_traj':
                    logits_traj = output[key]
                    probs_traj = fcn(logits_traj)
                    probs_traj = probs_traj    
                single_output[key] = (output[key][b]).cpu().detach().numpy()
            predictions = single_output['predictions']
            probs_occ = probs_occ[b]
            probs_traj = probs_traj[b]
        else:
            predictions = None
            probs_occ = None
            probs_traj = None

        save_name = f'{single_batch["id"]}-{name}_b_{b}'
        visualize(single_batch, predictions, 
                  probs_occ, probs_traj, 
                  plot_future, handpicked_anchors,
                  visualize_ids, os.path.join(vis_path,save_name),
                  just_polygon, just_anchors, visibility_only, 
                  save, fig, ax, xlim, ylim)

def visualize(sample, predictions, 
               probs_occ, probs_traj,
               plot_future, handpicked_anchors, 
               visualize_ids=False, filename='outputs/test.png', 
               just_polygon=False, just_anchors=False, visbility_only=False, 
               save=True, fig=None, ax=None, xlim=None, ylim=None):
    
    if ax is None:
        fig = plt.figure(num=1, clear=True, figsize=(10,10))
        ax = fig.add_subplot()
    else:
        ax.clear()

    obs = sample['observed_trajectories']
    future = sample['labels']
    polylines = sample['polylines']
    anchors = sample['anchors']
    polygon_corners = sample['polygon_corners']
    num_occluded_anchors = (sample['num_occluded_anchors']).astype(np.int32).item()
    ego_idx = sample['ego_idx']
    org_obs = sample['org_obs']
    org_future = sample['org_labels']
    sample_id = sample['id']

    if 'all_polygon_corners' in sample.keys():
        all_polygon_corners = sample['all_polygon_corners']
    else:
        all_polygon_corners = None

    visualize_maps(polylines, ax)
    visualize_original_agents(org_obs, org_future, ego_idx, num_occluded_anchors, ax, plot_future, visualize_ids)
    visualize_agents(obs, future, ego_idx, num_occluded_anchors, ax, plot_future, visualize_ids)
    if not visbility_only:
        if plot_future and predictions is not None and probs_traj is not None and probs_occ is not None:
            visualize_predictions_all(predictions, future, anchors, probs_traj, probs_occ, ax, handpicked_anchors, num_occluded_anchors)
        else:
            visualize_occlusion(anchors, future, polygon_corners, ax, probs_occ, num_occluded_anchors, just_anchors, just_polygon, all_polygon_corners=all_polygon_corners)
            pass

    if visbility_only and just_polygon:
        visualize_occlusion(anchors, future, polygon_corners, ax, probs_occ, num_occluded_anchors, just_anchors, just_polygon, all_polygon_corners=all_polygon_corners)

    ax.scatter(0,0, marker='*', s=300, color='darkorange', zorder=7) 

    if xlim is not None and ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_ylim(-60, 60)
        ax.set_xlim(-60, 60)

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if save:
        plt.savefig(f'{filename}_{just_polygon}_{just_anchors}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.savefig(f'{filename}_{just_polygon}_{just_anchors}.png', dpi=300, bbox_inches='tight', pad_inches=0.0)

        fig.clf()
        plt.close(fig)
    else:
        plt.show()


def visualize_agents(obs, future, ego_idx, num_occluded_anchors, ax, plot_future, visualize_ids=False): 
    ego_idx = int(ego_idx[0])  
    visualize_obs(ax, obs, future, ego_idx, size=100)
    if plot_future:
        visualize_future(ax, obs, future, num_occluded_anchors, ego_idx, traj_color='red')

def visualize_original_agents(obs, future, ego_idx, num_occluded_anchors, ax, plot_future, visualize_ids=False):
    ego_idx = int(ego_idx[0])         
    visualize_obs(ax, obs, future, ego_idx, traj_color='grey', rec_color='dimgrey', edge_color='grey', arrow_color='grey', zorder=2.5, size=60, alpha_rec=1.0)
    if plot_future:
        visualize_future(ax, obs, future, num_occluded_anchors, ego_idx, traj_color='grey', size=5)

def visualize_predictions_all(predictions, labels, anchors, probs_traj, probs_occ, ax, handpicked_anchors=[], num_occluded_anchors=None):
    if len(predictions.shape) == 4:
        N, M, T, _ = predictions.shape
        gt_anchors_indices = labels[:,0,-1]
        valid = ~np.isnan(anchors[:,0]) 

        if len(handpicked_anchors) > 0:
            valid[:num_occluded_anchors] = False
            valid[handpicked_anchors] = True

        for n in range(N):
            if valid[n]:
                if np.isnan(anchors[n,0]).any() or np.isnan(anchors[n,1]).any():
                    continue
                if n < num_occluded_anchors and len(handpicked_anchors) == 0: 
                    continue #Only visualize traj for real objects.
                
                for m in range(M):
                    traj = predictions[n,m]
                    alpha = (probs_traj[n,m]).item() #.numpy()
                    ax.plot(traj[:,0], traj[:,1], linewidth=10, alpha=alpha, color='deepskyblue',zorder=2.5)
        for n in range(N):
            if valid[n]:         
                if n >= num_occluded_anchors:
                    ax.scatter(anchors[n,0], anchors[n,1], linewidth=1, s=300, marker='o', color='red', edgecolors='black', zorder=500)
                else:
                    ax.scatter(anchors[n,0], anchors[n,1], linewidth=1, s=300, marker='o', color='grey', edgecolors='black', zorder=2)

def visualize_occlusion(anchors, labels, polygon_corners, 
                        ax, probs=None, num_occluded_anchors=np.inf, 
                        just_anchors=False, just_polygon=False, all_polygon_corners=None):

    points_data = []
    N = anchors.shape[0]
    if not just_polygon:
        ax.add_patch(Polygon(polygon_corners, color='grey', fill=True, alpha=0.3, linewidth=3.0))
        valid = ~np.isnan(anchors[:,0]) 
        if probs is not None:
            for idx in range(anchors.shape[0]):
                if valid[idx]:
                    if idx < num_occluded_anchors:
                        if just_anchors:
                            ax.scatter(anchors[idx,0], anchors[idx,1], marker='o', s=10, alpha=1.0, color='red', edgecolors='none')
                        else:
                            p = (probs[idx, 1]).item() #.numpy()
                            if np.isnan(p) or np.isnan(anchors[idx,0]) or np.isnan(anchors[idx,1]):
                                raise ValueError(f'Nan value in probs: {p}, anchor: {anchors[idx]}')
                            points_data.append([anchors[idx,0], anchors[idx,1], p])
                            c = lighten_color('red', amount=p)
                            ax.scatter(anchors[idx,0], anchors[idx,1], marker='o', s=10, alpha=0.85, color=c, edgecolors='none', zorder=10)
                        #ax.text(anchors[idx,0], anchors[idx,1], f'{idx}', fontsize=3, color='black', clip_on=True)

            points_data = np.array(points_data)        
        else:
            ax.add_patch(Polygon(polygon_corners, color='grey', alpha=0.6, fill=True))
            for n in range(N):
                if valid[n]:   
                    if n >= num_occluded_anchors:
                        ax.scatter(anchors[n,0], anchors[n,1], linewidth=1, s=10, marker='o', color='red', edgecolors='black', zorder=100)
                    else:
                        ax.scatter(anchors[n,0], anchors[n,1], linewidth=1, s=10, marker='o', color='red', edgecolors='black', zorder=100, alpha=0.4)
    else:
        if all_polygon_corners is not None:
            for single_polygon in all_polygon_corners:
                ax.add_patch(Polygon(single_polygon, color='grey', edgecolor='black', alpha=0.5, fill=True, zorder=-10))
        else:
            ax.add_patch(Polygon(polygon_corners, color='grey', alpha=0.6, fill=True))

def visualize_maps(polylines, ax):
    for poly_idx in range(polylines.shape[0]):
        valid = ~np.isnan(polylines[poly_idx,:,0])
        if not np.any(valid):
            continue
        map_type = np.argwhere(polylines[poly_idx, 0, 4:] == 1)[0][0]
        marker, color, size = colors_num[map_type]
        polyline = polylines[poly_idx,:,:2]
        valid = ~np.isnan(polyline[:,0]) # != np.nan
        if map_type in [2,4,5]:
            ax.plot(polyline[valid][:,0], polyline[valid][:,1], color=color, linewidth=size, alpha=0.6)
        elif map_type != 3:
            ax.scatter(polyline[valid][:,0], polyline[valid][:,1], marker=marker, s=size, color=color, alpha=0.6)
        else:
            ax.add_patch(Polygon(polyline[valid], fill=False, edgecolor='grey', alpha=0.6))

def plot_vehicle_rectangle(ax, obs, valid_agent, ego_idx, t_idx, patch_fc='black',
                        patch_edge='black', arrow_fc='black', arrow_edge='black', alpha_rec=1.0):
    heading = obs[ego_idx, t_idx, HEADING]
    length = valid_agent[t_idx, LENGTH]
    width = valid_agent[t_idx, WIDTH]
    angle = valid_agent[t_idx, HEADING] 
    veh_size = (width, length)
    pos = (valid_agent[t_idx,0], valid_agent[t_idx,1])

    pos = rotate_around_point_highperf(pos, -(heading - 90)*np.pi/180, (0,0))
    pos_bl = (pos[0] - veh_size[0]/2, pos[1] - veh_size[1]/2)
    pos_bl = rotate_around_point_highperf(pos_bl, (90 - angle - heading)*np.pi/180, pos)
    pos_bl = rotate_around_point_highperf(pos_bl, (heading - 90)*np.pi/180, (0,0)) 
    # plot a heading with arrow if needed
    facing_angle = angle + 90
    # ax.arrow(org_pos[0], org_pos[1], vel_x, vel_y,
    #             head_width=0.5, head_length=0.5, width=0.0005, fc=arrow_fc, ec=arrow_edge, zorder=3.5)

    ax.add_patch(Rectangle((pos_bl[0],pos_bl[1]), veh_size[0], veh_size[1], facing_angle - 90, 
                            alpha=alpha_rec, ec=patch_edge, fc=patch_fc, zorder=2)) 
                           
def visualize_obs(ax, obs_data, future, ego_idx, 
                  traj_color='blue', 
                  rec_color='black',
                  edge_color='black',
                  arrow_color='orange',
                  alpha_rec=1.0,
                  zorder=3, size=1):
    num_agents = obs_data.shape[0]
    for agent_idx in range(num_agents):
        agent = obs_data[agent_idx]
        anchor_idx = future[agent_idx,0,-1]
        valid = ~np.isnan(agent[:,0])
        valid_agent = agent[valid]

        if valid_agent.shape[0] == 0:
            continue
                  
        length = valid_agent[-1, LENGTH] 
        width = valid_agent[-1, WIDTH]
        angle = valid_agent[-1, HEADING] 

        if np.isnan(angle).any() or np.isnan(length).any() or np.isnan(width).any():
            continue

        ax.scatter(valid_agent[:,0], valid_agent[:,1], marker='.', alpha=1.0, s=size, color=traj_color, zorder=zorder)
        ax.plot(valid_agent[:,0], valid_agent[:,1], color=traj_color, linewidth=4, zorder=zorder)
        ax.plot(valid_agent[:,0], valid_agent[:,1], color=traj_color, linewidth=3, zorder=zorder)

        if valid_agent.shape[0] != 0:
            if np.isnan(anchor_idx) or np.isnan(agent[-1,:]).any():
                continue 

            agent_type = agent[-1,-1]
            if np.isnan(agent_type).any() or agent_type == -1:
                continue

            ax.scatter(valid_agent[-1,0], valid_agent[-1,1], marker='.', s=1*size, color=traj_color, zorder=zorder)            
            plot_vehicle_rectangle(ax, obs_data, valid_agent, ego_idx, -1, patch_fc=rec_color, patch_edge=edge_color, arrow_fc=arrow_color, arrow_edge=arrow_color, alpha_rec=alpha_rec)

def visualize_future(ax, obs_data, future_data, num_occluded_anchors, ego_idx, 
                    traj_color='blue', 
                    zorder=3.5, size=5):
    num_agents = future_data.shape[0]
    for agent_idx in range(num_agents):

        # If object is observed, just plot the future trajectory.
        # If object is not observed before, plot the future trajectory + the obj's rectangle.
        if np.isnan(future_data[agent_idx,0,-1]):
            continue
        anchor_idx = int(future_data[agent_idx,0,-1])
        agent = future_data[agent_idx]
        valid = ~np.isnan(agent[:,0]) 
        valid_agent = agent[valid]

        if valid_agent.shape[0] == 0:
            # All future is NaNs. Skip
            continue

        if np.isnan(valid_agent[0,:]).any():
            continue

        ax.scatter(valid_agent[:,0], valid_agent[:,1], marker='.', alpha=1.0, s=size, color=traj_color, zorder=zorder)
        ax.plot(valid_agent[:,0], valid_agent[:,1], color=traj_color, linewidth=4, zorder=zorder)
    
def lighten_color(color, amount=0.5):
    import colorsys

    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
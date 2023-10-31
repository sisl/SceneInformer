from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader

def compute_loss(output: Dict, sample: Dict, configs: Dict, metrics_data_return: bool = False) -> Tuple[torch.Tensor, Dict]:
    predictions = output['predictions']
    logits_traj = output['logits_traj']
    logits_occ = output['logits_occ']

    anchors = sample['anchors'].float()
    labels = sample['labels'].float()

    B, N, K, T, D = predictions.shape
    device = predictions.device
    
    #############  1. Get anchors #############
    gt_anchors = labels[:,:,0,-1] 
    anchor_indices = torch.argwhere(~torch.isnan(gt_anchors)) 
    label_indices = anchor_indices.clone() 
    anchor_indices[:,1] = gt_anchors[anchor_indices[:,0], anchor_indices[:,1]] 

    ############# 2. Get masks #############
    # If the anchor is invalid (nan, not in a map, etc), it is set to 0.0.
    invalid_anchors_mask = torch.ones(B, N).to(device)   
    invalid_anchors = torch.argwhere(torch.isnan(anchors[:,:,0])) 
    invalid_anchors_mask[invalid_anchors[:,0], invalid_anchors[:,1]] = 0.0

    # If anchor is used (appears in labels), it is set to 1.0. There are valid anchors that are not used in occlusions. 
    used_anchors_mask = torch.zeros(B, N).to(device)
    used_anchors_mask[anchor_indices[:,0], anchor_indices[:,1]] = 1.0 

    ############# 3. We populate gt_trajectory with all labels in the correct positins. Some of them could be nans #############
    gt_trajectory = torch.zeros((B,N,1,T,2)).to(device) 
    gt_trajectory[anchor_indices[:,0], anchor_indices[:,1], :, :, :] = (labels[label_indices[:,0], label_indices[:,1],:,0:2]).unsqueeze(1)
    
    gt_trajectory[invalid_anchors[:,0], invalid_anchors[:,1]] = 0   
    true_valid_labels = ~torch.isnan(gt_trajectory)
    gt_trajectory = torch.nan_to_num(gt_trajectory, nan=0) 
    
    gt_valid_anchor = used_anchors_mask * invalid_anchors_mask 
    gt_valid_mask = true_valid_labels * used_anchors_mask[:,:,None, None, None] * invalid_anchors_mask[:,:,None, None, None] 
    sample['gt_valid_mask'] = gt_valid_mask
    pred_trajs = predictions.reshape(B*N,K,T,D) 
    gt_trajs = gt_trajectory.reshape(B*N,1,T,2) 
    gt_valid_mask = gt_valid_mask.reshape(B*N,1,T,2)
    gt_valid_anchor = gt_valid_anchor.reshape(B*N, 1) 
    pred_scores = torch.softmax(logits_traj, dim=-1).reshape(B*N,K)
    regression_loss, regression_indices = nll_loss_gmm_direct(pred_scores, pred_trajs, gt_trajs, gt_valid_mask, gt_valid_anchor)
    regression_indices = regression_indices.reshape(B,N)
    regression_loss = regression_loss.reshape(B,N)
    regression_loss = regression_loss.mean(1)
    ############################################################################################
    

    ############# 5. Evaluate classification loss ############# 
    targets_traj = torch.zeros((B,N)).long().to(device)
    targets_occ = torch.zeros((B,N)).long().to(device)

    targets_traj[anchor_indices[:,0], anchor_indices[:,1]] = regression_indices[anchor_indices[:,0], anchor_indices[:,1]] 
    targets_occ[anchor_indices[:,0], anchor_indices[:,1]] = 1 
    targets_occ = targets_occ.reshape(B*N)
    targets_traj = targets_traj.reshape(B*N)
    logits_traj = logits_traj.reshape(B*N, -1)
    logits_occ = logits_occ.reshape(B*N, -1)

    occ_weights = torch.ones(logits_occ.shape[1]).to(device)
    occ_weights[1] = configs['entropy_weight']

    occ_entropy_loss_fcn = torch.nn.CrossEntropyLoss(weight=occ_weights, reduction='none')
    occ_entropy_loss = occ_entropy_loss_fcn(logits_occ, targets_occ).to(device) #(B,N)

    traj_entropy_loss_fcn = torch.nn.CrossEntropyLoss(reduction='none')
    traj_entropy_loss = traj_entropy_loss_fcn(logits_traj, targets_traj).to(device) #(B,N)

    occ_entropy_loss = occ_entropy_loss.reshape(B,N)
    traj_entropy_loss = traj_entropy_loss.reshape(B,N)

    entropy_mask = torch.ones_like(occ_entropy_loss).to(device)

    entropy_mask[invalid_anchors[:,0], invalid_anchors[:,1]] = 0.0
    occ_entropy_loss *= entropy_mask
    traj_entropy_loss *= entropy_mask

    occ_entropy_loss = occ_entropy_loss.mean((1)) #(B,)
    traj_entropy_loss = traj_entropy_loss.mean((1)) #(B,)

    total_loss = (configs['reg_const'] * regression_loss + \
                configs['occ_class_const'] * occ_entropy_loss + \
                configs['traj_class_const'] * traj_entropy_loss).mean(0) 

    metrics_dict = {
        'total_loss': total_loss,
        'regression_loss': regression_loss.mean(),
        'occ_entropy_loss': occ_entropy_loss.mean(),
        'traj_entropy_loss': traj_entropy_loss.mean(),
    }

    if metrics_data_return:
        metrics_data = {
            'predictions': pred_trajs,
            'gt_trajectory': gt_trajs,
            'gt_valid_mask': gt_valid_mask,
        }

        return total_loss, metrics_dict, metrics_data
    else:
        return total_loss, metrics_dict

def nll_loss_gmm_direct(pred_scores, pred_trajs, gt_trajs, gt_valid_mask, gt_valid_anchor, pre_nearest_mode_idxs=None,
                        timestamp_loss_weight=None, use_square_gmm=True, log_std_range=(-1.609, 1.0), rho_limit=0.5): #log_std_range=(-1.609, 1.0)
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi 

    Args:
        pred_scores (batch_size, num_modes):
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
    """
    batch_size = pred_scores.shape[0]

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs)
        distance = (distance * gt_valid_mask)
        distance = distance.norm(dim=-1) 
        distance = distance.sum(dim=-1)  
        nearest_mode_idxs = distance.argmin(dim=-1)
    nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  

    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  
    res_trajs = (gt_trajs[:, 0, :, :] - nearest_trajs[:, :, 0:2]) * gt_valid_mask[:, 0, :, :] 
    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    if use_square_gmm:
        log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        std1 = std2 = torch.exp(log_std1)   # (0.2m to 150m)
        rho = torch.zeros_like(log_std1)
    else:
        log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)  
        std2 = torch.exp(log_std2)  
        rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    gt_valid_anchor = gt_valid_anchor.type_as(pred_scores)
    gt_valid_mask = gt_valid_mask.type_as(pred_scores) #fixme: where is pred_scores?
    if timestamp_loss_weight is not None:
        gt_valid_anchor = gt_valid_anchor * timestamp_loss_weight[None, :]

    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # (batch_size, num_timestamps)
    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask[:, 0, :, 0]).mean(dim=-1) * gt_valid_anchor[:, 0] 
    
    return reg_loss, nearest_mode_idxs



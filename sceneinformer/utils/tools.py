import numpy as np
import torch
import math

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler, batch_size=self.batch_size)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
            
def save_model(model, optimizer, path, epoch):
    torch.save(model.state_dict(), f'{path}/model_{epoch}.pth')
    torch.save(optimizer.state_dict(), f'{path}/optimizer_{epoch}.pth')

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
        points = [np.array([np.nan, np.nan])]
    return points

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

def convert_cartesian_point_to_polar(point):
    x, y = point
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.array([r, theta]) #(-pi, pi)

def check_if_point_is_in_cone(point, p1, p2):
    # TODO: But beyond the cone, the point is not visible.

    #Convert to polar coordinates
    point = convert_cartesian_point_to_polar(point)
    p1 = convert_cartesian_point_to_polar(p1)
    p2 = convert_cartesian_point_to_polar(p2)

    min_theta = min(p1[1], p2[1])
    max_theta = max(p1[1], p2[1])

    # Check point.
    if point[0] <= p1[0] or point[0] <= p2[0]:
        return False
    
    # Check angle.
    if min_theta < -np.pi/2 and max_theta > np.pi/2:
        if point[1] > max_theta:
            return True
        elif point[1] < min_theta:
            return True
        return False
    else:
        #Check if point is in cone
        if min_theta <= point[1] <= max_theta:
            return True
        else:
            return False

# From nanoGPT by Andrej Karpathy
def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
import torch

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
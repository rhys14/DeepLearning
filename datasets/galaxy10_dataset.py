import h5py
import torch
from torch.utils.data import Dataset

class Galaxy10Dataset(Dataset):
    def __init__(self, fp):
        self.file_path = fp
        self.file = None
        
        with h5py.File(fp, "r") as f:
            self.length = len(f["images"])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")

        image = self.file["images"][idx]
        label = self.file["ans"][idx]
        
        image = torch.from_numpy(image.copy()).float()
        
        #rearragnes images into right order for channel orignally is Height width channel
        #instead of Channel Height Width
        image = image.permute(2,0,1) / 255.0
        
        label = torch.tensor(int(label), dtype=torch.long)
        
        return image, label


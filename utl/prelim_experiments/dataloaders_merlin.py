from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import pydicom as dicom
import numpy as np

class PBIImageSet(Dataset):
    def __init__(self, transform=None, dset=None):
        self.truth_df = pd.read_excel('.xlsx') #excel file with class information
        self.img_dir = '/' #image dir containing .npy volumes of CT scans
        self.transform = transform
        self.datalist = #list of case IDs
        self.datalist.sort()
        
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        
        label = #get label for single case
        
        image = #get image for single case
        
        output = {"image": image, "label": label}
        
        if self.transform:
            output = self.transform(output)
        
        return output

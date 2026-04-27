import numpy as np
from torch.utils.data import Dataset
class SpeedDataset(Dataset):
    
    def __init__(self, flow_array_path:str, labels_path:str, window_size:int, is_train:bool=True):
        # load flow array
        try:
            self.loadAndSplit(flow_array_path,labels_path)
            self.window_size = window_size
            self.is_train = is_train
            
        except Exception as e:
            print(f'Failedto load data set : {e}')

        pass
    
    def loadAndSplit(self, data_path:str, lables_path:str, train_split:float=0.8):
        
        tempData = np.load(data_path)
        
        tempLables = np.loadtxt(lables_path)

        frames:int = tempData.shape[0]
        train_split_count = int(frames*train_split)
        self.train: np.ndarray = tempData[:train_split_count]
        self.train_lables: np.ndarray = tempLables[:train_split_count]
        self.verify: np.ndarray = tempData[train_split_count:]
        self.verify_lables: np.ndarray = tempLables[train_split_count:]
    

    def __getitem__(self, idx) -> tuple[np.ndarray, np.float32]:
        if self.is_train:
            return (self.train[idx: idx+self.window_size],self.train_lables[idx+self.window_size])
        else:
            return (self.verify[idx: idx+self.window_size],self.verify_lables[idx+self.window_size])
    
    def __len__(self):
        if self.is_train:
            return self.train.shape[0] - self.window_size
        else:
            return self.verify.shape[0] - self.window_size
        



# test
# dataset = SpeedDataset("../data/precomputed_flows.npy", "../data/train.txt", 20)

# print(dataset.train.shape)
# print(dataset.verify.shape)
# print(dataset.__getitem__(100)[0].shape)
# print(dataset.__getitem__(100)[1])
# print(dataset.__len__())
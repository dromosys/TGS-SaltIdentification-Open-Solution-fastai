from fastai.dataset import *

class CustomDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
        
    def get_x(self, i): 
        return open_image(os.path.join(self.path,self.fnames[i]))
    def get_y(self, i): 
        return open_image(os.path.join(self.path,self.y[i]))
    def get_c(self): return 0
    
class DepthDataset(Dataset):
    def __init__(self,ds,dpth_dict):
        self.dpth = dpth_dict
        self.ds = ds
        
    def __getitem__(self,i):
        val = self.ds[i]
        return val[0],self.dpth[self.ds.fnames[i].split('/')[2][:-4]],val[1]
    
class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
        
    def get_x(self, i): 
        return open_image(os.path.join(self.path, self.fnames[i]))
    
    def get_y(self, i):
        return open_image(os.path.join(str(self.path), str(self.y[i])))

    def get_c(self): return 0
    
class TestFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform,flip, path):
        self.y=y
        self.flip = flip
        super().__init__(fnames, transform, path)
        
    def get_x(self, i): 
        im = open_image(os.path.join(self.path, self.fnames[i]))
        return np.fliplr(im) if self.flip else im
        
    def get_y(self, i):
        im = open_image(os.path.join(str(self.path), str(self.y[i])))
        return np.fliplr(im) if self.flip else im
    def get_c(self): return 0

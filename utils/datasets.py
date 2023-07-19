from torch.utils.data import Dataset 
from torch.utils.data import DataLoader, ConcatDataset

import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import spectral as spec
import torch

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)])
img_transform

class Img(Dataset):
    def __init__(self, img_df, _3d=True, transform=img_transform):
        self.img_df = img_df
        self.transform = transform
        self._3d = _3d

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        img_path = self.img_df['path'].iloc[idx]
        img = spec.envi.open(img_path)
        # np_img = np.array(img[:, :, 5:60])
        label = self.img_df['class'].iloc[idx]
        dim = img.shape
        
        if self._3d:
            image = torch.zeros((1, 55, dim[0], dim[1]), dtype=torch.float)
            for i, c in enumerate(range(5, 60)):
                image[:,i,:,:] = self.transform(img[:,:,c])
        
        else:
            image = torch.zeros((55, dim[0], dim[1]), dtype=torch.float)
            for i, c in enumerate(range(5, 60)):
                image[i,:,:] = self.transform(img[:,:,c])
        
        return image, label
    
def wrapPatch(patch):
    if type(patch) == list:
        return [Img(img_df = patch[i]) for i in range(2)]
    else:
        return Img(img_df = patch)
    
def tensorToImg(tensor, img_size, rgb_bands=[7, 15, 32]):
    img = torch.zeros(img_size, img_size, len(rgb_bands))
    for i, b in enumerate(rgb_bands):
        img[:,:,i] = tensor[b]
    return img

def displayImgs(imgs, labels, classes, n, img_size=20, save=False):
    if n > 32:
        n = 32
    fig = plt.figure(figsize=(20,11))
    for i, img in enumerate(imgs[:n]):
        img = tensorToImg(img[0], img_size)
        fig.add_subplot(4, 8, i + 1).title.set_text(classes[labels[i]])
        plt.imshow(img)
    plt.subplots_adjust(wspace=0.2)
    plt.show()
    if save:
        fig.savefig('fig.png')
        
def getTransform(i, n):
    d = int(360/(n + 1)) * (i + 1)
    print(f"\t Rotated {d} deg")
    return transforms.Compose([transforms.ToTensor(),
                        transforms.RandomRotation(degrees=(d, d)),
                        transforms.ConvertImageDtype(torch.float)])

def transformDs(patch, n=1):
    ds = []
    for i in range(n):
        ds.append(Img(img_df=patch, transform=getTransform(i, n)))
    
    return ds

def augmentPatch(patch, aug_com, n):
    patch_ds_list = [[],[]]
    dataset = [[],[]]
    for i in range(len(aug_com)):
        patch_ds_c = patch.getDataset([aug_com[i] if i2 == i else 0 for i2 in range(len(aug_com))])
        for i2 in range(2):
            patch_ds_list[i2].append(patch_ds_c[i2])
            dataset_c = transformDs(patch_ds_c[i2], n[i])
            for dc in dataset_c:
                dataset[i2].append(dc)
                
    return dataset, patch_ds_list

def augmentPatch2(img, n, verbose=["Training set", "Validation set"]):
    dataset = [[] for i in verbose]
    if type(img) != list:
        img = [img]
    for i, d in enumerate(verbose):           
        print(d)
        for i2 in range(len(n)):
            if n[i2] > 0:
                print(f'class index: {i2}')
            img_c = img[i].img_df[img[i].img_df['class'] == i2]
            img_c = img_c.sample(frac=1)
            dataset_c = transformDs(img_c, n[i2])
            for dc in dataset_c:
                dataset[i].append(dc)
    
    if len(dataset) == 1:
        dataset = dataset[0]
    
    return dataset 

def countImg(img, classes):
    c = {c:0 for c in classes}
    for i in img:
        c[classes[i[1]]] += 1
    
    print(f"{len(img)}, with {c}")
    

def imagesLoader(imgs, batch_size):
    loaders = []
    for i in range(len(imgs)):
        
        img = imgs[i][0]
        shuffle = imgs[i][1]
        
        if type(img) == list:
            img = ConcatDataset(img)
            
        loaders.append(DataLoader(dataset=img, batch_size=batch_size, shuffle=shuffle))
        
    return loaders
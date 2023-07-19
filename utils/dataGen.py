import os
import pandas as pd
import numpy as np

class Patches():
    def __init__(self, cwd, classes=["background", "leaf", "diseased"]):
        self.cwd = cwd
        self.avail_types = ["hdr", "jpg"]
        self.c = classes
        self.images_df = pd.DataFrame(self.__populateImg(), columns=["path", "type", "class"])
        
    def __populateImg(self):
        images = []
        img_type_path = self.cwd
        # patches_path = self.cwd
        # for image in os.listdir(patches_path):
        #     img_type_path = patches_path + "\\" + image
        for img_type in os.listdir(img_type_path):
            if img_type in self.avail_types:
                class_path = img_type_path + "\\" + img_type
                for _class in os.listdir(class_path):
                    img_dir = class_path + "\\" + _class
                    for img in os.listdir(img_dir):
                        img_path = img_dir + "\\" + img
                        if img[-3:] != "img":
                            images.append([img_path, img_type, _class])
        return images
    
    def describe(self, ignore='jpg'):
        for type_ in self.avail_types:
            if ignore != type_:
                counts = self.images_df[self.images_df["type"] == type_]["class"].value_counts()
                if counts is not None:
                    print(f"For {type_} image, there are: \n", counts, "\n")
    
    def digitize(self, d, df):
        dup = True
        for i, c in enumerate(self.c):
            df.loc[(df['class'] == c), 'class'] = i
            
        u = df.duplicated(subset=['path'])
        if True in list(u):
            dup = False
            
        print(f'{d} has {len(df)} samples {df["class"].unique()} class indices included')
        
        return dup
    
    def generateDataset(self, dc, data_type='hdr', seed=0):
        dataset_composition = [[self.c[i], dc[i]] for i in range(len(self.c))]
        patch_ds = pd.DataFrame(columns=["path", "type", "class"])
        np.random.seed(seed)
        
        for dc in dataset_composition:
            ids = self.images_df.loc[(self.images_df['class'] == dc[0]) & 
                                     (self.images_df['type'] == data_type)].index
            patch_ds = pd.concat(
                [patch_ds, self.images_df.iloc[np.random.choice(ids, dc[1], replace=False)]])
        
        dup = self.digitize('Dataset', patch_ds)
        
        print(f'   Example: {patch_ds.iloc[0]["path"]}')
        print(f'   Unique: {dup}')
        print('\n')
        
        return patch_ds   
    
    def getDataset(self, dc, data_type='hdr', train_frac = .7, seed=0):
        dataset_composition = [[self.c[i], dc[i]] for i in range(len(self.c))]
        train_df = pd.DataFrame(columns=["path", "type", "class"])
        val_df = train_df.copy()
        np.random.seed(seed)
        
        for dc in dataset_composition:
            ids = self.images_df.loc[(self.images_df['class'] == dc[0]) & 
                                     (self.images_df['type'] == data_type)].index
            temp_df = self.images_df.iloc[np.random.choice(ids, dc[1], replace=False)]
            temp_train = temp_df.sample(frac=train_frac, replace=False, random_state=1)
            temp_val = temp_df.drop(temp_train.index)

            train_df = pd.concat([train_df, temp_train])
            val_df= pd.concat([val_df, temp_val])
        
        patch_ds = [train_df, val_df]
        
        dup = True
        for d, p in zip(["Training set", "Validation set"], patch_ds):
            dup = self.digitize(d, p)
        
        print(f'   Example: {p.iloc[0]["path"]}')
        print(f'   Unique: {dup}')
        print('\n')

        return patch_ds
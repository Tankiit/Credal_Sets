import numpy as np
import os
from glob import glob
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms

class AnimalDataset(data.Dataset):
    def __init__(self, classes_file, root_dir='data', transform=None):
        # Load the continuous predicate matrix (50 classes x 85 attributes)
        predicate_path = os.path.join(root_dir, 'predicate-matrix-continuous.txt')
        self.predicate_mat = np.array(np.genfromtxt(predicate_path, dtype='float'))
        
        # Load predicate (attribute) names
        self.predicate_names = []
        with open(os.path.join(root_dir, 'predicates.txt')) as f:
            for line in f:
                self.predicate_names.append(line.strip())

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

        # Build class name to index mapping from train/test files
        self.class_to_index = {}
        index = 0
        # Read both train and test files to get all classes
        for split_file in ['trainclasses.txt', 'testclasses.txt']:
            with open(os.path.join(root_dir, split_file)) as f:
                for line in f:
                    class_name = line.strip()
                    if class_name not in self.class_to_index:
                        self.class_to_index[class_name] = index
                        index += 1

        # Load all image paths and their class indices for this split
        self.img_names = []
        self.img_index = []
        with open(os.path.join(root_dir, classes_file)) as f:
            for line in f:
                class_name = line.strip()
                FOLDER_DIR = os.path.join(root_dir, 'JPEGImages', class_name)
                file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
                files = glob(file_descriptor)

                class_index = self.class_to_index[class_name]
                for file_name in files:
                    self.img_names.append(file_name)
                    self.img_index.append(class_index)

    def __getitem__(self, index):
        # Load and transform image
        im = Image.open(self.img_names[index])
        if im.getbands()[0] == 'L':
            im = im.convert('RGB')
        if self.transform:
            im = self.transform(im)

        # Get class index and attribute vector
        im_index = self.img_index[index]
        im_predicate = self.predicate_mat[im_index,:]
        
        return {
            'features': im,
            'concepts': torch.FloatTensor(im_predicate),
            'label': torch.LongTensor([im_index])[0],
            'image_path': self.img_names[index]
        }

    def __len__(self):
        return len(self.img_names)

    def get_concept_names(self):
        """Return list of attribute names"""
        return self.predicate_names
    
    def get_class_names(self):
        """Return list of class names in index order"""
        return [k for k, _ in sorted(self.class_to_index.items(), key=lambda x: x[1])]

if __name__ == '__main__':
  dataset = AnimalDataset('testclasses.txt')
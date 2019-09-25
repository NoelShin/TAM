import os
from glob import glob
from csv import reader
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomCrop, RandomHorizontalFlip, RandomResizedCrop
from torchvision.transforms import Resize, ToTensor
from PIL import Image


class CustomCIFAR10(Dataset):
    def __init__(self, opt, val=False):
        super(CustomCIFAR10, self).__init__()
        dir_dataset = opt.dir_dataset

        if val:
            self.dataset = CIFAR10(root=dir_dataset, train=False, download=True)
            self.transform = Compose([ToTensor(),
                                      Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262])])

        else:
            self.dataset = CIFAR10(root=dir_dataset, train=True, download=True)
            self.transform = Compose([RandomCrop((32, 32), padding=4, fill=0, padding_mode='constant'),
                                      RandomHorizontalFlip(),
                                      ToTensor(),
                                      Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262])])

    def __getitem__(self, index):
        return self.transform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


class CustomCIFAR100(Dataset):
    def __init__(self, opt, val=False):
        super(CustomCIFAR100, self).__init__()
        dir_dataset = opt.dir_dataset

        if val:
            self.dataset = CIFAR100(root=dir_dataset, train=False, download=True)
            self.transform = Compose([ToTensor(),
                                      Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])

        else:
            self.dataset = CIFAR100(root=dir_dataset, train=True, download=True)
            self.transform = Compose([RandomCrop((32, 32), padding=4, fill=0, padding_mode='constant'),
                                      RandomHorizontalFlip(),
                                      ToTensor(),
                                      Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])

    def __getitem__(self, index):
        return self.transform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


class CustomImageNet1K(Dataset):
    def __init__(self, opt, val=False):
        super(CustomImageNet1K, self).__init__()
        dir_dataset = opt.dir_dataset
        self.dir_input = sorted(glob(os.path.join(dir_dataset, 'Val' if val else 'Train', '*.JPEG')))
        if val:
            path_label = os.path.join('./ILSVRC2012_validation_ground_truth.txt')
            label = list()
            with open(path_label, 'r') as txt_file:
                for row in txt_file:
                    label.append(int(row) - 1)
            self.label = label

            self.transform = Compose([Resize(256),
                                      CenterCrop(224),
                                      ToTensor(),
                                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        else:
            path_label = os.path.join('./training_WNID2class.txt')
            dict_WNID2label = dict()
            with open(path_label, 'r') as txt_file:
                csv_file = reader(txt_file, delimiter=',')
                for i, row in enumerate(csv_file):
                    if i != 0:
                        dict_WNID2label.update({row[0]: int(row[1]) - 1})  # -1 is for making the label start from 0.
                    else:
                        pass
            self.label = dict_WNID2label

            self.transform = Compose([RandomResizedCrop(224),
                                      RandomHorizontalFlip(),
                                      ToTensor(),
                                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.val = val

    def __getitem__(self, index):
        path_image = self.dir_input[index]
        if self.val:
            return self.transform(Image.open(path_image).convert('RGB')), self.label[index]

        else:
            WNID = os.path.splitext(os.path.split(path_image)[-1])[0][:9]
            return self.transform(Image.open(path_image).convert('RGB')), self.label[WNID]

    def __len__(self):
        return len(self.dir_input)


class CustomSVHN(Dataset):
    def __init__(self, opt, val=False):
        super(CustomSVHN, self).__init__()
        dir_dataset = opt.dir_dataset

        if not val:
            self.dataset = SVHN(root=dir_dataset, split='train', download=True)
            self.transform = Compose([ToTensor()])

        else:
            self.dataset = SVHN(root=dir_dataset, split='test', download=True)
            self.transform = Compose([ToTensor()])

    def __getitem__(self, index):
        return self.transform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)

# code partially from https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/cifar.py
from torchvision import transforms

from .randaugment import RandAugmentMC


class TransformFixMatch(object):
    '''
    Perform Strong and Weak transformation (except RandomErasing) as described in FixMatch paper

    Return
        tuple: (normalized weak transformation, normalized strong transformation)
    '''
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.Resize((224, 224))])

        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.Resize((224, 224)),
            RandAugmentMC(n=2, m=10)])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

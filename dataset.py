import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision

"""数据加载器"""


class FaceDataset(Dataset):

    def __init__(self, size=32, path='data'):
        super(FaceDataset, self).__init__()
        self.dataset = []
        for image_name in os.listdir(path):
            image_path = f'{path}/{image_name}'
            self.dataset.append(image_path)
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size + int(0.15 * size)),
                torchvision.transforms.RandomCrop(size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image_path = self.dataset[item]
        image = Image.open(image_path)
        image_tensor = self.train_transform(image)
        return image_tensor


if __name__ == '__main__':
    face = FaceDataset()
    a = face[0]
    print(a.max(), a.min())

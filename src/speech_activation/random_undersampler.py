import os
import random
import torch
from torch.utils.data import Sampler
from torchvision.datasets import ImageFolder

class CustomRandomUndersampler(Sampler):
    def __init__(self, data_source: ImageFolder):

        self.data_source: ImageFolder = data_source
        self.classes_ = self._get_classes()
        self.classes_counts = {k: len(v) for k, v in self.classes_.items()}
        self.num_samples = min(self.classes_counts.values()) * len(self.classes_)


    def _get_classes(self):
        classes = dict()
        for i, (_, label) in enumerate(self.data_source.imgs):
            label = self.data_source.classes[label]
            if label not in classes:
                classes[label] = []
            classes[label].append(i)

        return classes


    def __iter__(self):
        indices = []
        min_count = min(self.classes_counts.values())
        for c in self.classes_:
            class_indices = self.classes_[c]
            random_indices = random.sample(class_indices, min_count)
            indices.extend(random_indices)

        random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as T
    import matplotlib.pyplot as plt

    # Example usage
    data_source = ".\\data\\silence_dataset\\valid"  # Replace with your dataset path
    ds = ImageFolder(data_source, transform=T.ToTensor())  # Replace with your dataset class
    sampler = CustomRandomUndersampler(ds)
    loader = DataLoader(ds, sampler=sampler, batch_size=1)

    print("Class counts:")
    print(sampler.classes_counts)  # Print the class counts
    print(sampler.classes_.keys())

    for sample in sampler:
        print(sample)  # Print the sampled data
        break

    data, target = next(iter(loader))
    plt.imshow(data[0].permute(1, 2, 0))  # Display the first image in the batch
    plt.title(f"Label: {target[0]}")
    plt.show()

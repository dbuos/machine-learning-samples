
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import ViTFeatureExtractor
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip
from torch.utils.data import DataLoader


def load_eurosat_for_vit(image_path = './data/eurosat', batch_size = 128, num_workers=20):
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    transform = Compose([
        RandomResizedCrop(feature_extractor.size),
        RandomHorizontalFlip(),
        ToTensor(), 
        Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])

    eurosat_dataset = datasets.EuroSAT(root=image_path, transform=transform, download=True)

    #Randomize the dataset
    torch.manual_seed(1)
    train_len = int(0.85 * len(eurosat_dataset)) - int(0.85 * len(eurosat_dataset) * 0.2)
    valid_len = int(0.85 * len(eurosat_dataset) * 0.2)
    test_len = len(eurosat_dataset) - train_len - valid_len
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(eurosat_dataset, [train_len, valid_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    return loaders
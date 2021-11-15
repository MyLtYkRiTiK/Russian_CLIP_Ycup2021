import numpy as np
import seaborn as sns
import torch
import torch.utils.data as loader
import torchvision
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from tqdm.autonotebook import tqdm

from src.clip.clip import tokenize
from src.clip.model import CLIP

BICUBIC = InterpolationMode.BICUBIC


def convert_image_to_rgb(image):
    return image.convert("RGB")


def _preprocess(n_px=224):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        convert_image_to_rgb,
        ToTensor( ),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def del_funk(v, k):
    del v[k]
    return v


def inference(model, testloader, classes):
    text_tokens = tokenize(classes)
    text_tokens = text_tokens.to('cuda:0')
    text_features = model.encode_text(text_tokens).float( )
    text_features /= text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad( ):
        for images, labels in tqdm(testloader):
            images = images.to('cuda:0')
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
            predictions = [int(x) for x in torch.argmax(similarity, dim=-2)]
            total += labels.size(0)
            correct += (predictions == labels.numpy( )).sum( ).item( )
            all_labels.extend(labels)
            all_preds.extend(predictions)
    print(f'Accuracy of the network on the {len(all_preds)} test images: %d %%' % (
            100 * correct / total))
    return all_labels, all_preds


def main():
    preprocess = _preprocess( )
    checkpoint = torch.load('path_to_file/epoch_30_light.pt',
                            map_location="cuda:0")
    sd = checkpoint["state_dict"]
    if next(iter(sd.items( )))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items( )}
    model_info = {
        "embed_dim": 512,
        "image_resolution": 224,
        "vision_layers": 12,
        "vision_width": 768,
        "vision_patch_size": 32,
        "context_length": 77,
        "vocab_size": 69856,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12
    }
    model = CLIP(**model_info)
    model.load_state_dict(sd)
    model.eval( ).requires_grad_(False).to('cuda:0')
    for param in model.parameters( ):
        param.grad = None

    batch_size = 100

    for dataset in ['caltech101', 'cifar10', 'cifar100']:
        if dataset == 'caltech101':
            plt.figure(figsize=(30, 20))
            testset = datasets.ImageFolder(root='./data/caltech101/101_ObjectCategories/', transform=preprocess)
            testloader = loader.DataLoader(testset, batch_size=batch_size, shuffle=False)
        elif dataset == 'cifar10':
            plt.figure(figsize=(20, 10))
            testset = torchvision.datasets.CIFAR10(root='./data',
                                                   download=True, transform=preprocess)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=False, num_workers=10)
        elif dataset == 'cifar100':
            plt.figure(figsize=(40, 30))
            testset = torchvision.datasets.CIFAR100(root='./data',
                                                    download=True, transform=preprocess)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=False, num_workers=10)

        with open(f'./data/{dataset}_classes.txt', 'r') as f:
            classes_labels = [x.strip( ) for x in f.readlines( )]
        print(classes_labels)
        print(len(classes_labels))

        labels, preds = inference(model, testloader, classes_labels)
        labels = [classes_labels[x] for x in labels]
        preds = [classes_labels[x] for x in preds]

        cm = confusion_matrix(labels, preds, labels=classes_labels)
        sns_plot = sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                               xticklabels=classes_labels, yticklabels=classes_labels)
        sns_plot.figure.savefig(f"{dataset}.jpg")
        plt.show( )


if __name__ == "__main__":
    main( )

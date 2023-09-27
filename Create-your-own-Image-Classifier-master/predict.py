# import numpy as np
# import argparse
# import torch
# from torch import nn, optim
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
# from torch.autograd import Variable
# from collections import OrderedDict
# import matplotlib.pyplot as plt
# import json
# from PIL import Image
# import futility
# import fmodel

# parser = argparse.ArgumentParser(description = 'Parser for predict.py')

# parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
# parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
# parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
# parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
# parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
# parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

# args = parser.parse_args()
# path_image = args.input
# number_of_outputs = args.top_k
# device = args.gpu
# json_name = args.category_names
# path = args.checkpoint

# def main():
#     model=fmodel.load_checkpoint(path)
#     with open(json_name, 'r') as json_file:
#         name = json.load(json_file)
        
#     probabilities = fmodel.predict(path_image, model, number_of_outputs, device)
#     probability = np.array(probabilities[0][0])
#     labels = [name[str(index + 1)] for index in np.array(probabilities[1][0])]
    
#     i = 0
#     while i < number_of_outputs:
#         print("{} with a probability of {}".format(labels[i], probability[i]))
#         i += 1
#     print("Finished")

    
# if __name__== "__main__":
#     main()



import argparse
import numpy as np
import json
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, default='train_checkpoint.pth')
    parser.add_argument('image_path', metavar='image_path', type=str, default='flowers/test/58/image_02663.jpg')
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--top_k', action='store', dest="top_k", type=int, default=5)
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    _model = getattr(torchvision.models, checkpoint['pretrained_model'])(pretrained=True)
    _model.input_size = checkpoint['input_size']
    _model.output_size = checkpoint['output_size']
    _model.learning_rate = checkpoint['learning_rate']
    _model.hidden_units = checkpoint['hidden_units']
    _model.learning_rate = checkpoint['learning_rate']
    _model.classifier = checkpoint['classifier']
    _model.epochs = checkpoint['epochs']
    _model.load_state_dict(checkpoint['state_dict'])
    _model.class_to_idx = checkpoint['class_to_idx']
    _model.optimizer = checkpoint['optimizer']
    return _model


def process_image(image):
    resize = 256
    crop_size = 224
    (width, height) = image.size

    if height > width:
        height = int(max(height * resize / width, 1))
        width = int(resize)
    else:
        width = int(max(width * resize / height, 1))
        height = int(resize)


    im = image.resize((width, height))

    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size
    im = im.crop((left, top, right, bottom))

    im = np.array(im)
    im = im / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    im = np.transpose(im, (2, 0, 1))
    return im


def predict(image_path, model, top_k, gpu):
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif gpu and not(torch.cuda.is_available()):
        print("GPU was selected as the training device, but no GPU is available. Using CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    model.to(device)

    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image)
    image = image.unsqueeze_(0)
    image = image.float()

    with torch.no_grad():
        output = model.forward(image.cuda())

    p = F.softmax(output.data, dim=1)

    top_p = np.array(p.topk(top_k)[0][0])

    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(index_to_class[each]) for each in np.array(p.topk(top_k)[1][0])]

    return top_p, top_classes, device


def load_names(category_names_file):
    with open(category_names_file) as file:
        category_names = json.load(file)
    return category_names


def main():
    args = parse_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    model = load_checkpoint(checkpoint)

    top_p, classes, device = predict(image_path, model, top_k, gpu)

    category_names = load_names(category_names)

    labels = [category_names[str(index)] for index in classes]

    print(f"Results for your File: {image_path}")
    print(labels)
    print(top_p)
    print()

    for i in range(len(labels)):
        print("{} - {} with a probability of {}".format((i+1), labels[i], top_p[i]))


if __name__ == "__main__":
    main()
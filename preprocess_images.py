import torch
from torchvision.transforms import transforms
from utils import *
from math import ceil

train_annotations = read_csv_file('Annotations/processed_train_annotations.csv')
test_annotations = read_csv_file('Annotations/processed_test_annotations.csv')
val_annotations = read_csv_file('Annotations/processed_val_annotations.csv')
inceptionv3_size = 299

resnet50_size = 224

vgg16_size = 224

preprocess_inceptionv3 = transforms.Compose([
    transforms.Resize(inceptionv3_size),
    transforms.CenterCrop(inceptionv3_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_resnet50 = transforms.Compose([
    transforms.Resize(resnet50_size),
    transforms.CenterCrop(resnet50_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_vgg16 = transforms.Compose([
    transforms.Resize(vgg16_size),
    transforms.CenterCrop(vgg16_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_images(annotations, processor, processor_name, image_folder, eye, size):

    # extract label
    annotations = annotations[1:]
    len_images = len(annotations)
    max_size = 500
    count_of_parts = ceil(len_images / max_size)
    for part in range(count_of_parts):
        if(len_images >= (part+1)*max_size):
            images_tensor = torch.zeros(max_size, 3, size, size)
            start_point = part * max_size
            end_point = start_point + max_size
        else:
            tensor_size = len_images - ((part) * max_size)
            images_tensor = torch.zeros(tensor_size, 3, size, size)
            start_point = part * max_size
            end_point = start_point + tensor_size

        for ids, ann in tqdm(enumerate(annotations[start_point:end_point])):
            images_tensor[ids] = processor(gray_to_rgb(load_image(f"{image_folder}/{ann[0]}_{eye}.jpg")))

        torch.save(images_tensor, f"Preprocessed_Images/{processor_name}_{image_folder.lower()}_{eye}_part_{part}.pt")

def process_all_images(dataset_annotations, processors, processors_names, image_folders, eyes, sizes):

    for processor, processor_name, size in zip(processors, processors_names, sizes):

        for annotations, image_folder in zip(dataset_annotations, image_folders):

            for eye in eyes:
                print(f"{processor_name}, {image_folder}, {eye}")
                process_images(annotations=annotations, processor=processor, processor_name=processor_name, image_folder=image_folder, eye=eye, size=size)


dataset_annotations = [train_annotations, val_annotations, test_annotations]
processors = [preprocess_inceptionv3, preprocess_resnet50, preprocess_vgg16]
processors_names = ["inceptionv3", "resnet50", "vgg16"]
image_folders = ["Train", "Val", "Test"]
eyes = ["left", "right"]
sizes = [inceptionv3_size, resnet50_size, vgg16_size]

try:
    os.mkdir("Preprocessed_Images")
    print("Directory Preprocessed_Images Created ")
except FileExistsError:
    print("Directory Preprocessed_Images already exists")
process_all_images(dataset_annotations=dataset_annotations, processors=processors, processors_names=processors_names, image_folders=image_folders, eyes=eyes, sizes=sizes)


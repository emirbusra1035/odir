import torch
from torchvision.transforms import transforms
from utils import *
from math import ceil

train_annotations = read_csv_file('Annotations/processed_train_annotations.csv')
test_annotations = read_csv_file('Annotations/processed_test_annotations.csv')
val_annotations = read_csv_file('Annotations/processed_val_annotations.csv')

disease_classes = create_dict(train_annotations + val_annotations[1:] + test_annotations[1:])

# disease_classes = load_dict()
number_of_classes = len(disease_classes)


def create_target_tensor(annotations, eye, target_name):

    # extract label
    annotations = annotations[1:]
    len_images = len(annotations)
    max_size = 500
    count_of_parts = ceil(len_images / max_size)
    for part in range(count_of_parts):
        if(len_images >= (part+1)*max_size):
            targets_tensor = torch.zeros(max_size, number_of_classes)
            start_point = part * max_size
            end_point = start_point + max_size
        else:
            tensor_size = len_images - ((part) * max_size)
            targets_tensor = torch.zeros(tensor_size, number_of_classes)
            start_point = part * max_size
            end_point = start_point + tensor_size

        for ids, ann in tqdm(enumerate(annotations[start_point:end_point])):
            if eye == "left":
                for c in ann[5].split(','):
                    token = disease_classes[c]
                    targets_tensor[ids][token] = 1
            else:
                for c in ann[6].split(','):
                    token = disease_classes[c]
                    targets_tensor[ids][token] = 1

        torch.save(targets_tensor, f"Targets/{target_name.lower()}_{eye}_part_{part}.pt")

def create_all_targets(dataset_annotations, eyes, target_names):

    for annotations, target_name in zip(dataset_annotations, target_names):

        for eye in eyes:
            print(f"{target_name}, {eye}")
            create_target_tensor(annotations=annotations, target_name=target_name, eye=eye)


dataset_annotations = [train_annotations, val_annotations, test_annotations]
target_names = ["Train", "Val", "Test"]
eyes = ["left", "right"]
try:
    os.mkdir("Targets")
    print("Directory Targets Created ")
except FileExistsError:
    print("Directory Targets already exists")
create_all_targets(dataset_annotations=dataset_annotations, eyes=eyes, target_names=target_names)


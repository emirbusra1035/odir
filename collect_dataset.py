# collect the eye diseases as binary components.

from utils import *
from tqdm import tqdm

val_annotations = read_csv_file('Annotations/processed_val_annotations.csv')
val_annotations = val_annotations[1:]
eye_diseases = ["N", "D", "G", "C", "A", "H", "M", "O"]
eye_diseases_index = [-8, -7, -6, -5, -4, -3, -2, -1]

# Dataset -> {eye_diseases} -> 0 and 1 | Create folders of annotations for binary classification.
try:
    os.mkdir("Dataset_Val")
    print("Directory Dataset_Val Created ")
except FileExistsError:
    print("Directory Dataset_Val already exists")
for folder_name in eye_diseases:
    try:
        os.mkdir(f"Dataset_Val/{folder_name}")
        os.mkdir(f"Dataset_Val/{folder_name}/0")
        os.mkdir(f"Dataset_Val/{folder_name}/1")
        print(f"Directory {folder_name} Created ")
    except FileExistsError:
        print(f"Directory {folder_name} already exists")
# Move the images into binary folders.
for disease_class, disease_index in tqdm(zip(eye_diseases, eye_diseases_index)):
    print(disease_class)
    for ids, ann in tqdm(enumerate(val_annotations)):
        disease_class_of_person = int(ann[disease_index])
        person_id = int(ann[0])
        left_eye = gray_to_rgb(load_image(f'Val/{person_id}_left.jpg'))
        right_eye = gray_to_rgb(load_image(f'Val/{person_id}_right.jpg'))
        left_eye = resize(left_eye)
        right_eye = resize(right_eye)
        output_image = create_input_image(left_eye, right_eye)
        if disease_class_of_person:
            # save to 1 folder
            output_image.save(f'Dataset_Val/{disease_class}/1/{ids}.jpg')
        else:
            # save to 0 folder
            output_image.save(f'Dataset_Val/{disease_class}/0/{ids}.jpg')
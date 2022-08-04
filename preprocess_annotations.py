from utils import *
train_annotations = read_csv_file('Annotations/train.csv')
val_annotations = read_csv_file('Annotations/val.csv')
test_annotations = read_csv_file('Annotations/test.csv')

spoiled_keys = ["2174", "2175", "2176", "2177", "2178", "2179", "2180", "2181", "2182", "2957"]
spoiled_classes = ["anterior segment image", "no fundus image"]
trivial_classes = ["lens dust", "optic disk photographically invisible", "low image quality", "image offset"]
inappropriate_classes = spoiled_classes + trivial_classes
def check_data_appropriate(annotation):
    l = annotation.split(',')
    for c in l:
        if c in inappropriate_classes:
            return False
    return True


def check_person_valid(annotation):
    if annotation[0] in spoiled_keys:
        return False
    if not check_data_appropriate(annotation[5]):
        return False
    if not check_data_appropriate(annotation[6]):
        return False
    return True

def process_annotations(annotations, fn="processed_annotations"):

    processed_annotations = []
    processed_annotations.append(annotations[0])

    for ann in annotations[1:]:

        validity_of_person = check_person_valid(ann)
        if validity_of_person:
            processed_annotations.append(ann)
    
    with open(f"{fn}.csv", 'w') as f:
        write = csv.writer(f, delimiter=";")
        write.writerows(processed_annotations)

    return processed_annotations

process_annotations(annotations=train_annotations, fn="Annotations/processed_train_annotations")
process_annotations(annotations=val_annotations, fn="Annotations/processed_val_annotations")
process_annotations(annotations=test_annotations, fn="Annotations/processed_test_annotations")

def create_target_tensors(annotations):
    raise NotImplementedError

from utils import *
import torch
import torch.nn as nn
import csv
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
val_annotations = read_csv_file('/content/drive/MyDrive/oia-odir/Annotations/processed_test_annotations.csv')
val_annotations = val_annotations[1:]

models = ["vgg", "inception", "resnet"]
# models = ["vgg"]

softmax = nn.Softmax(dim=1)
for model in models:
    if model == "inception":
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    N = torch.load(f"/content/drive/MyDrive/oia-odir/oia-odir-best-models/{model}_N_1.pth", map_location=device)
    D = torch.load(f"/content/drive/MyDrive/oia-odir/oia-odir-best-models/{model}_D_1.pth", map_location=device)
    G = torch.load(f"/content/drive/MyDrive/oia-odir/oia-odir-best-models/{model}_G_1.pth", map_location=device)
    C = torch.load(f"/content/drive/MyDrive/oia-odir/oia-odir-best-models/{model}_C_1.pth", map_location=device)
    A = torch.load(f"/content/drive/MyDrive/oia-odir/oia-odir-best-models/{model}_A_1.pth", map_location=device)
    H = torch.load(f"/content/drive/MyDrive/oia-odir/oia-odir-best-models/{model}_H_1.pth", map_location=device)
    M = torch.load(f"/content/drive/MyDrive/oia-odir/oia-odir-best-models/{model}_M_1.pth", map_location=device)
    O = torch.load(f"/content/drive/MyDrive/oia-odir/oia-odir-best-models/{model}_O_1.pth", map_location=device)
    N.eval()
    D.eval()
    G.eval()
    C.eval()
    A.eval()
    H.eval()
    M.eval()
    O.eval()
    result_list = []
    labels = ['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    result_list.append(labels)
    for ann in tqdm(val_annotations):
        ann_id = ann[0]
        ann_left_path = ann[3]
        ann_right_path = ann[4]
        left_im = resize(gray_to_rgb(load_image(f"/content/Test/{ann_id}_left.jpg")))
        right_im = resize(gray_to_rgb(load_image(f"/content/Test/{ann_id}_right.jpg")))
        input_tensor = create_input_image(left_im, right_im)
        input_tensor = transform(input_tensor)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            N_result = softmax(N(input_tensor)).data[0, 1]
            D_result = softmax(D(input_tensor)).data[0, 1]
            G_result = softmax(G(input_tensor)).data[0, 1]
            C_result = softmax(C(input_tensor)).data[0, 1]
            A_result = softmax(A(input_tensor)).data[0, 1]
            H_result = softmax(H(input_tensor)).data[0, 1]
            M_result = softmax(M(input_tensor)).data[0, 1]
            O_result = softmax(O(input_tensor)).data[0, 1]
        result = [ann_id, N_result.item(), D_result.item(), G_result.item(), C_result.item(), A_result.item(), H_result.item(), M_result.item(), O_result.item()]
        result_list.append(result)

    with open(f'/content/drive/MyDrive/oia-odir/oia-odir-test-csv-results/{model}_processed_test_result_1.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result_list)
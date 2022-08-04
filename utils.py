import os
import sys
from tqdm import tqdm
import requests
from zipfile import ZipFile
import json
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import csv
import json
 
# returns csv data for the given path.
def read_csv_file(path):
 l = []
 with open(path, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for row in csv_reader:
        l.append(row)
    return l

# returns PIL image for the given path.
def load_image(path):
    return Image.open(path)

# shows the given input PIL image.
def show_image(img):
    img.show()

# returns rgb image of the given input image.
def gray_to_rgb(im):
    return im.convert('RGB')

# returns resized image for the given image.
def resize(im):
    return im.resize((299, 299))

# returns collected unique disease for the given annotations and save them into a json file.
def create_dict(annotations):
    d = {}
    index = 0
    for ann in annotations[1:]:
        left_eye = ann[5]
        right_eye = ann[6]
        splitted_left_eye = left_eye.split(',')
        splitted_right_eye = right_eye.split(',')
        for a in splitted_left_eye:
            if a not in d:
                d[a] = index
                index += 1
        for a in splitted_right_eye:
            if a not in d:
                d[a] = index
                index += 1
    with open('classes.json', 'w') as file:
        json.dump(d, file)
    return d

# returns saved disease classes from the json file.
def load_dict():
    with open('classes.json', 'r') as file:
        d = json.load(file)
    return d

# returns the mirrored image of the given image.
def mirror_image(im):
    return ImageOps.mirror(im)

# returns horizontal concatenation of the given two images.
def concat_horizontal(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
 
# returns vertical concatenation of the given two images.
def concat_vertical(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

# returns the input image for the deep learning architectures.
def create_input_image(im1, im2):
    concataneted = concat_horizontal(im1, im2)
    mirrored_conc = mirror_image(concataneted)
    output_image = concat_vertical(concataneted, mirrored_conc)
    return output_image
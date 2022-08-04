import Augmentor

paths = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
classifications = ['0', '1']

# Augment dataset for all given class images.
def augment(path, classification):
    pth = '/Users/bemir/Desktop/new/Dataset_Train/' + path + '/' + classification + '/'
    p = Augmentor.Pipeline(pth)
    p.random_distortion(probability=0.25, grid_width=3, grid_height=3, magnitude=3)
    p.gaussian_distortion(probability=0.25, grid_width=3, grid_height=3, magnitude=3, corner='bell', method='in')
    p.skew(probability=0.05)
    p.skew_tilt(probability=0.05)
    p.skew_left_right(probability=0.05)
    p.skew_top_bottom(probability=0.05)
    p.skew_corner(probability=0.05)
    p.sample(10000)


from tqdm import tqdm
for path in tqdm(paths):
    for classification in classifications:
        augment(path=path, classification=classification)
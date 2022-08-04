1  import logging
2  import logging.config
3  from os import listdir
4  from os.path import isfile, join
5  from image_crop import Crop
6  def process_all_images():
7  files = [f for f in listdir(source_folder)
8  if isfile(join(source_folder, f))]
9  for file in files:
10  logger.debug('Processing image: ' + file)
11  Crop(source_folder, destination_folder,
12  file).remove_black_pixels()
13  if __name__ == '__main__':
14  source_folder = r'/Users/bemir/ODIR-5K_Training_Dataset'
15  destination_folder = r'/Users/bemir/ODIR-5K_Training_Dataset_cropped'
16  # create logger
17  logging.config.fileConfig('logging.conf')
18  logger = logging.getLogger('odir')
19  process_all_images() 
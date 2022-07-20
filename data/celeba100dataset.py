import os
import csv
import shutil

os.mkdir("CelebA100")
path = 'CelebA/img_align_celeba_png'
with open("CelebA100.csv", newline='') as csvfile:
    rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    for row in rowreader:
        image_name = row[0]
        shutil.copy(os.path.join(path,image_name), os.path.join("CelebA100", image_name))

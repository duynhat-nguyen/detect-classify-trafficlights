from os import path
from glob import glob
from argparse import ArgumentParser
from shutil import copy2

parser = ArgumentParser(description='Copy some xml files')
parser.add_argument('--source', type=str, help='Soure path', required=True)
parser.add_argument('--destination', type=str, help='Destination path', required=True)

args = parser.parse_args()

image_paths = glob(args.destination + "/*")

for image_path in image_paths:
    copy2(args.source + "/" + path.splitext(path.basename(image_path))[0] + ".xml", args.destination)
    


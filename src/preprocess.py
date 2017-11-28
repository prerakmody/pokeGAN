# resize pokeGAN.py

import os
import cv2
import shutil
from PIL import Image
from tqdm import tqdm_notebook

def prepareData(cwd = 'os', data_dir = '../data', src_img = 'data', dest_resize = 'data_resized', dest_bw = 'data_resized_black' ,nx = 256, ny = 256):
    if cwd == 'os':
        cwd = os.getcwd()
    data_dir = os.path.join(cwd, data_dir)
    print ('[PREPROCESS] Data Dir:', data_dir)
    src_img = os.path.join(data_dir, src_img)
    print ('[PREPROCESS] Src Images:', src_img)
    dest_resize = os.path.join(data_dir, dest_resize)
    print ('[PREPROCESS] Dest Resized Images :', dest_resize)
    
    if os.path.exists(src_img):
        if os.path.exists(dest_resize):
            shutil.rmtree(dest_resize)
        if not os.path.exists(dest_resize):
            os.mkdir(dest_resize)

        # 1. Resize Images
        images = os.listdir(src_img)
        with tqdm_notebook(total = len(images), desc='Resize') as pbar:
            for image in images:
                pbar.update(1)
                img = cv2.imread(os.path.join(src_img,image))
                img = cv2.resize(img,(nx,ny))
                cv2.imwrite(os.path.join(dest_resize,image), img)

        # 2. Convert to B/W
        src_resize = dest_resize
        dest_bw = os.path.join(data_dir, dest_bw)
        if os.path.exists(dest_bw):
            shutil.rmtree(dest_bw)
        if not os.path.exists(dest_bw):
            os.mkdir(dest_bw)

        images_resized = os.listdir(src_resize)
        with tqdm_notebook(total = len(images_resized), desc='B/W') as pbar:    
            for image in images_resized:
                pbar.update(1)
                # img = cv2.imread(os.path.join(src_resize,image))
                png = Image.open(os.path.join(src_resize,image))
                if png.mode == 'RGBA':
                    png.load() # required for png.split()
                    background = Image.new("RGB", png.size, (0,0,0))
                    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
                    background.save(os.path.join(dest_bw, image.split('.')[0] + '.jpg'), 'JPEG')
                else:
                    png.convert('RGB')
                    png.save(os.path.join(dest_bw, image.split('.')[0] + '.jpg'), 'JPEG')
    
    else:
        print (' No images present in the "{0}" directory'.format(src_img))

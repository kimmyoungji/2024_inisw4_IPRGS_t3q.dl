import os
from PIL import Image, ImageDraw
import re
import numpy as np


def process(output, cropped_car, seg_image, car_bbox, reader):
    
    if 'error' in output:
        return None, None, output['error']
    
    if len(output) == 0:
        return None, None, "can't detect license plate"
    
    bbox = output[0]['box'].values()
    xmin, ymin, xmax, ymax = map(int, bbox)
    cropped_plate = cropped_car.crop((xmin, ymin, xmax, ymax))

    output = reader.readtext(np.asarray(cropped_plate))
    
    license_number = ""
    for (_, text, _) in output:
        text = re.sub(r'[^가-힣0-9]', '', text)
        license_number += text

    car_xmin, car_ymin, car_xmax, car_ymax = car_bbox
    new_xmin = car_xmin + xmin
    new_ymin = car_ymin + ymin
    new_xmax = car_xmin + xmax
    new_ymax = car_ymin + ymax

    draw = ImageDraw.Draw(seg_image, 'RGBA')
    draw.rectangle([(new_xmin, new_ymin), (new_xmax, new_ymax)], outline="green", width=2)
    
    seg_lp_image = seg_image

    return seg_lp_image, license_number, None
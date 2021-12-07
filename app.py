from flask import Flask, request, make_response, render_template, redirect, url_for

#General libraries
import re, cv2, os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import random
import decimal
import opendatasets as od
import keras
import math
import scipy

from deskew import determine_skew
from typing import Tuple, Union

import segmentation_models as sm
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import warnings
warnings.filterwarnings("ignore")

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

app = Flask(__name__)

#image_folder = os.path.join(os.getcwd(), 'images')
#app.config["UPLOAD_FOLDER"] = image_folder

segmentation_model_file = 'final_segmentation_model'
faster_rcnn_path = 'output/model_final.pth' #<-- for cfg.MODEL.WEIGHTS

def prod_resize_input(img_link):
    '''
    Function takes an image and resizes it.
    '''
    img = cv2.imread(img_link)
    img = cv2.resize(img, (224, 224))
    return img.astype('uint8')

#Create function to crop images.
def crop_for_seg(img, bg, mask):
    '''
    Function extracts an image where it overlaps with its binary mask.
    img - Image to be cropped.
    bg - The background on which to cast the image.
    mask - The binary mask generated from the segmentation model.
    '''
    #mask = mask.astype('uint8')
    fg = cv2.bitwise_or(img, img, mask=mask)
    fg_back_inv = cv2.bitwise_or(bg, bg, mask=cv2.bitwise_not(mask))
    New_image = cv2.bitwise_or(fg, fg_back_inv)
    return New_image

def extract_meter(image_to_be_cropped):
    '''
    Function further extracts image such that the meter reading takes up the majority of the image.
    The function finds the edges of the ROI and extracts the portion of the image that contains the entire ROI.
    '''
    where = np.array(np.where(image_to_be_cropped))
    x1, y1, z1 = np.amin(where, axis=1)
    x2, y2, z2 = np.amax(where, axis=1)
    sub_image = image_to_be_cropped.astype('uint8')[x1:x2, y1:y2]
    return sub_image

def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    '''
    This function attempts to rotate meter reading images to make them horizontal.
    Its arguments are as follows:

    image - The image to be deskewed (in numpy array format).
    angle - The current angle of the image, found with the determine_skew function of the deskew library.
    background - The pixel values of the boarder, either int (default 0) or a tuple.

    The function returns a numpy array.
    '''
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def resize_aspect_fit(img, final_size: int):
    '''
    Function resizes the image to specified size.

    path - The path to the directory with images.
    final_size - The size you want the final images to be. Should be in int (will be used for w and h).
    write_to - The file you wish to write the images to.
    save - Whether to save the files (True) or return them.
    '''
    im_pil = Image.fromarray(img)
    size = im_pil.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im_pil = im_pil.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im_pil, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    new_im = np.asarray(new_im)
    return np.array(new_im)

def prep_for_ocr(img):
    img = resize_aspect_fit(img, 224)
    output_name = 'test_img.jpg'
    cv2.imwrite(output_name, img)
    return output_name

#Segment input image.
def segment_input_img(img):

    #Resize image.
    img_small = prod_resize_input(img)

    #Open image and get dimensions.
    input_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    input_w = int(input_img.shape[1])
    input_h = int(input_img.shape[0])
    dim = (input_w, input_h)

    #Load model, preprocess input, and obtain prediction.
    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    img_small = preprocess_input(img_small)
    img_small = img_small.reshape(-1, 224, 224, 3).astype('uint8')
    model = tf.keras.models.load_model(segmentation_model_file, custom_objects={'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss, 'iou_score' : sm.metrics.iou_score})
    mask = model.predict(img_small)

    #Change type to uint8 and fill in holes.
    mask = mask.astype('uint8')
    mask = scipy.ndimage.morphology.binary_fill_holes(mask[0, :, :, 0]).astype('uint8')

    #Resize mask to equal input image size.
    mask = cv2.resize(mask, dsize=dim, interpolation=cv2.INTER_AREA)

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((10,10), np.uint8)

    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    #Create background array.
    bg = np.zeros_like(input_img, 'uint8')

    #Get new cropped image and make RGB.
    New_image = crop_for_seg(input_img, bg, mask)
    New_image = cv2.cvtColor(New_image, cv2.COLOR_BGR2RGB)

    #Extract meter portion.
    extracted = extract_meter(New_image)

    grayscale = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)

    if angle == None:
        angle = 1

    rotated = rotate(extracted, angle, (0, 0, 0))
    return rotated

def get_reading(image_path):
    '''
    This is the main function for the pipeline.
    It takes an input image path as its only argument.
    It then carries out all the necessary steps to extract a meter reading.
    The output is the reading.

    NOTE: Due to having to load and generate predictions from two models,
    this script may take a while to run.
    '''

    #Segment image.
    segmented = segment_input_img(image_path)

    #Prep image and save path.
    prepped_path = prep_for_ocr(segmented)

    #Class labels.
    labels = ['number', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    #List for storing meter readings.
    list_of_img_reading = []

    #Configure model parameters.
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = 'output/model_final.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
    predictor = DefaultPredictor(cfg)

    #Read prepped image and obtain prediction.
    im = cv2.imread(prepped_path)
    outputs = predictor(im)

    #Find predicted boxes and labels.
    instances = outputs['instances']
    coordinates = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    pred_classes = outputs['instances'].pred_classes.cpu().tolist()

    #Obtain list of all predictions and the leftmost x-coordinate for bounding box.
    pred_list = []
    for pred, coord in zip(pred_classes, coordinates):
        pred_list.append((pred, coord[0]))

    #Sort the list based on x-coordinate in order to get proper order or meter reading.
    pred_list = sorted(pred_list, key=lambda x: x[1])

    #Get final order of identified classes, and map them to class value.
    final_predictions = [x[0] for x in pred_list]
    pred_class_names = list(map(lambda x: labels[x], final_predictions))

    #Add decimal point to list of digits depending on number of bounding boxes.
    if len(pred_class_names) == 5:
        pass
    else:
        pred_class_names.insert(5, '.')

    #Combine digits and convert them into a float.
    combine_for_float = "".join(pred_class_names)
    meter_reading = float(combine_for_float)

    return meter_reading

@app.route('/', methods=['GET', 'POST'])
def hello_world():

    if request.method == 'GET':
        return render_template('index.html', value='hi')

    if request.method == 'POST':
        imagefile = request.files['file']
        image_path = os.path.join(os.getcwd(), imagefile.filename)
        imagefile.save(image_path)

        reading = get_reading(image_path)
        return render_template('result.html', meter_reading=reading)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

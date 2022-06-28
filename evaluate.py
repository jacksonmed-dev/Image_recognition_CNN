import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from Mask_RCNN.mrcnn.model import MaskRCNN
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import model as modellib, utils

# Path to trained weights file
current_path = os.getcwd()
MODEL_DIR = '/content/drive/MyDrive/logs/body parts20220627T2341/mask_rcnn_body parts_0050.h5' # change your file path here

############################################################
#  Configurations
############################################################


class FoodConfig(Config):
    # Give the configuration a recognizable name
    NAME = "body parts"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class FoodDataset(utils.Dataset):

    def load_food(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        # self.add_class("food", 1, "Chilli_Chicken")
        # self.add_class("food", 2, "Tandoori_Chicken")
        # self.add_class("food", 3, "Gulab_Jamun")
        # self.add_class("food", 4, "Ice_Cream")
        # self.add_class("EleCom", 1, "one")
        # self.add_class("EleCom", 2, "two")
        # self.add_class("EleCom", 3, "three")
        # self.add_class("EleCom", 4, "four")
        # self.add_class("EleCom", 5, "five")
        # self.add_class("EleCom", 6, "six")
        # self.add_class("EleCom", 7, "seven")
        # self.add_class("EleCom", 8, "eight")
        # self.add_class("EleCom", 9, "nine")
        # self.add_class("EleCom", 0, "zero")
        # self.add_class("EleCom", 11, "component")
        # self.add_class("EleCom", 12, "nike")
        self.add_class("body parts", 1, "head")
        self.add_class("body parts", 2, "shoulder")
        self.add_class("body parts", 3, "buttocks")
        self.add_class("body parts", 4, "leg")
        self.add_class("body parts", 5, "arm ")
        self.add_class("body parts", 6, "heel")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "component.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.

            polygons=[]
            objects=[]
            for r in a['regions']:
                polygons.append(r['shape_attributes'])

                objects.append(r['region_attributes'])
            class_ids =[]
            for n in objects:
                #print(n)
                if n['body parts'] == "head":
                    class_ids.append(1)
                elif n['body parts'] == "shoulder":
                    class_ids.append(2)
                elif n['body parts'] == "buttocks":
                    class_ids.append(3)
                elif n['body parts'] == "leg":
                    class_ids.append(4)
                elif n['body parts'] == "arm":
                    class_ids.append(5)
                elif n['body parts'] == "heel":
                    class_ids.append(6)
            #class_ids = [n['body parts'] for n in objects]

            
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "body parts",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
				class_ids=class_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "body parts":
            return super(self.__class__, self).load_mask(image_id)
        class_ids = image_info['class_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        #class_ids=np.array([self.class_names.index(shapes[0])])
        #print("info['class_ids']=", info['class_ids'])
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids#[mask.shape[-1]] #np.ones([mask.shape[-1]], dtype=np.int32)#class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "body parts":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 6
     
dataset = FoodDataset()
dataset.load_food('/content/drive/MyDrive/dataset_body/dataset_body', "val")
dataset.prepare()   

weights_path = MODEL_DIR
# Load weights
model = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

from Mask_RCNN.mrcnn import utils

config=FoodConfig()
total_gt = np.array([]) 
total_pred = np.array([]) 
mAP_ = [] #mAP list

#compute total_gt, total_pred and mAP for each image in the test dataset
# Compute total ground truth boxes(total_gt) and total predicted boxes(total_pred) and mean average precision for each Image 
#in the test dataset
for image_id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =modellib.load_image_gt(dataset, config, image_id)#, #use_mini_mask=False)
    info = dataset.image_info[image_id]

    # Run the model
    print(len(image))
    results = model.detect([image], verbose=0)
    r = results[0]
    
    #compute gt_tot and pred_tot
    gt, pred = utils.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
    total_gt = np.append(total_gt, gt)
    total_pred = np.append(total_pred, pred)
    
    #precision_, recall_, AP_ 
    AP_, precision_, recall_, overlap_ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'])
    #check if the vectors len are equal
    print("the actual length of the ground truth vect is : ", len(total_gt))
    print("the actual length of the predicted vect is : ", len(total_pred))
    
    mAP_.append(AP_)
    print("Average precision of this image : ",AP_)
    print("The actual mean average precision for the whole images", sum(mAP_)/len(mAP_))


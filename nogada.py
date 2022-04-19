import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import copy

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)



from imantics import Polygons, Mask

from glob import glob

# a = sorted(list(glob('/data/sglee/crawling/download1/*/*.jpg')))
# a = sorted(list(glob("/data/sglee/crawling_2/AutoCrawler/download/far away person image/*.jpg")))
a = sorted(list(glob("/data/download_se/*/*.jpg")))

for im_path in a:
    
    im = cv2.imread(im_path)
    outputs = predictor(im)
    outputs = outputs['instances'][outputs['instances'].pred_classes==0]
    
    H,W,_ = im.shape
    h = 0
    w = 0
    if H/480 > W/640:
        h = 480
        w = int(W*480/H)
    else:
        h = int(H*640/W)
        w = 640
    
    ori_im = np.zeros((480,640,3))
    im = cv2.resize(im, (w,h))
    ori_im[:h,:w,:] = im
    im_vis = copy.deepcopy(ori_im)
    
    labels = []
    valid = False
    for idx in range(len(outputs)):
        output = outputs[idx]
        label = {"label": "person", "group_id": 'null', "line_color": None, "fill_color": None}
        points = Mask(output.pred_masks[0].cpu().detach().numpy()).polygons().points[0]
        
        points = points.astype(np.float32)
        points[:,0] = points[:,0]*float(w)/W
        points[:,1] = points[:,1]*float(h)/H
        

        label['points'] = points.tolist()
        label['group_id'] = idx+1
        label['shape_type'] = "polygon"
        label['flags'] = {}
        labels.append(label)

        bbox = output.pred_boxes.tensor[0]
        area = ((bbox[2]-bbox[0])*float(w)/W)*((bbox[3]-bbox[1])*float(h)/H)
        if area < 80*80:
            valid = True
            
        im_vis = cv2.polylines(im_vis, [points.astype(np.int64).reshape(-1,1,2)], isClosed=False, color=(0,255,255))

    num_people = len(outputs)
    # count the number of people
    if len(outputs) == 0 or len(outputs) > 10:
        continue
    if valid == False:
        continue

    # cv2.imshow('test', im_vis)
    # cv2.waitKey()
    image_path = '-'.join(im_path.split('/')[-2:])
    anno_dict = {
        "version": "3.16.7",
        "flags": {},
        "shapes": labels,
        "imagePath": image_path,
        "imageData": None,
        "lineColor": [
            0,
            255,
            0,
            128
        ],
        "fillColor": [
            255,
            0,
            0,
            128
        ],
        "imageHeight": 480,
        "imageWidth": 640
    }
    if num_people > 5:
        num_people = 5
    elif num_people > 1:
        num_people = 2
    elif num_people == 1:
        num_people = 1
    
    with open(f'/data/download_se/results/{num_people}/'+image_path.replace('jpg','json'), 'w') as f:
        json.dump(anno_dict, f)

    cv2.imwrite(f'/data/download_se/results_viz/{num_people}/'+ image_path, im_vis)
    cv2.imwrite(f'/data/download_se/results/{num_people}/'+ image_path, ori_im)

    # cv2.imwrite(out.get_image()[:, :, ::-1], 'results/'+im_path.split('/')[-1])
    # cv2.waitKey(0)
import re
import json
import os

import torch
import torch.distributed as dist

import utils
from PIL import Image, ImageFilter, ImageDraw

from tqdm import tqdm


def blur_except_box(image, bounding_box):
    """
    Apply blur to all areas except the selected bounding box.

    :param image_path: Path to the input image.
    :param bounding_boxes: List of bounding boxes in the format [(x1, y1, x2, y2), ...].
    :param output_path: Path to save the output image.
    """
    # Apply blur filter to the entire image
    blurred_image = image.filter(ImageFilter.GaussianBlur(15))

    # Create a mask for the unblurred area
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw the selected bounding box on the mask
    draw.rectangle(bounding_box, fill=255)

    # Composite the images using the mask
    final_image = Image.composite(image, blurred_image, mask)

    return final_image


def get_captions(annotation):
    data = []
    id = 0
    for i, record in tqdm(enumerate(annotation), total=len(annotation)):
        image_id = next(iter(record.keys()))
        image_path = image_id+".jpg"
        
        for region in record[image_id]['regions']:
            bbox = (region['x'], region['y'], region['x']+region['width'], region['y']+region['height'])
        
            for caption in region['captions'][:1]:
                item = {
                    'prompt': "This image is that ",
                    'caption': caption['caption'],
                    'image': image_path,
                    'image_id': image_id,
                    'bbox': bbox,
                    'id': id,
                }
                data.append(item)
                id += 1
    return data

def get_each_json(ann_paths):
    annotation = []
    for ann_path in ann_paths:
        data = json.load(open(ann_path, "r"))
        if type(data) == dict:
            annotation.append(data)
    return annotation


def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file



from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

from data.utils import get_each_json, get_captions

from pathlib import Path
import glob

def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval


def core_caption_eval(coco_gt_root, results_file, split):
    annotation = get_each_json(glob.glob(str(Path(coco_gt_root) / 'test/*json')))
    annotation = get_captions(annotation)

    # annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO()
    coco.dataset = annotation
    coco.createIndex()

    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval
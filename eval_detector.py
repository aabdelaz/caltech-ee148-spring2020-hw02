import os
import json
import numpy as np

def point_inside_box(point, box):
    x_min = box[0]
    x_max = box[2]
    y_min = box[1]
    y_max = box[3]
    
    if point[0] < x_max and point[0] > x_min\
        and point[1] > y_min and point[1] < y_max:
            return True
    return False            


def box_inside_box(box_1, box_2):
    '''Determines if box_1 is inside box_2'''
    min_point = box_1[:2]
    max_point = box_1[2:] 
    if point_inside_box(min_point, box_2) and\
        point_inside_box(max_point, box_2):
            return True
    return False    

def corners(box):
    x = [box[0], box[2]]*2
    y = [box[1]]*2 +  [box[3]]*2
    return list(zip(x,y))

def box_intersects_box(box_1, box_2):    
    c1 = corners(box_1)
    for corner in c1:
        if point_inside_box(corner, box_2):
            return True

    c2 = corners(box_2)
    for corner in c2:
        if point_inside_box(corner, box_1):
            return True
    
    return False

def corners_inside(box_1, box_2):
    c1 = corners(box_1)
    b2_list = [box_2] * 4
    corners_inside = list(map(point_inside_box, c1, b2_list))
    return corners_inside

def area(box):
    return (box[2] - box[0])*(box[3] - box[1])
    
def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    if not box_intersects_box(box_1, box_2):
        return 0.0
    if box_inside_box(box_1, box_2):
        iou =  area(box_1)/area(box_2)
    if box_inside_box(box_2,box_1):
        iou = area(box_2)/area(box_1)
    
    c_bools = corners_inside(box_1, box_2)
    if sum(c_bools) == 1:
        c1 = corners(box_1)
        c2 = corners(box_2)
        # Figure out which corner
        diff = []
        if c_bools[0]:
            # 0th corner of box 1 is in box 2
            diff = [a - b for a, b in zip(c1[0], c2[3])]
        elif c_bools[1]:
           diff = [a - b for a, b in zip(c1[1], c2[2])] 
        elif c_bools[2]:
            diff = [a - b for a, b in zip(c1[2], c2[1])] 
        else:
            diff = [a - b for a, b in zip(c1[3], c2[0])]
        I = abs(diff[0]*diff[1])
        U = area(box_1) + area(box_2) - I
        iou = I/U
    
    # At this point, we must have exactly 2 points of one box inside the other
    # If box 2 has two corners in box 1, swap the names so that the subsequent
    # code (which assumes that box_1 has 2 points inside box 2) still works.
    if sum(c_bools) == 0:
        c_bools = corners_inside(box_2, box_1)
        tmp = box_1;
        box_1 = box_2
        box_2 = tmp
    else:
        assert(sum(c_bools) == 2)
    
    if c_bools[0] and c_bools[1]:
        I = abs((box_1[2] - box_1[0])*(box_2[3] - box_1[1]))
    elif c_bools[1] and c_bools[2]: 
        I = abs((box_1[3] - box_1[1])*(box_2[0] - box_1[2]))
    elif c_bools[2] and c_bools[3]:
        I = abs((box_1[2] - box_1[0])*(box_2[1] - box_1[3]))
    else:
        I = abs((box_1[3] - box_1[1])*(box_2[2] - box_1[0]))
    U = area(box_1) + area(box_2) - I
    iou = I/U
    
    assert (iou >= 0) and (iou <= 1.0)
    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.iteritems():
        gt = gts[pred_file]
        for i in range(len(gt)):
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])


    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = './data/hw02_preds'
gts_path = './data/hw02_annotations'

# load splits:
split_path = './data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 


confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)

# Plot training set PR curves

if done_tweaking:
    print('Code for plotting test set PR curves.')

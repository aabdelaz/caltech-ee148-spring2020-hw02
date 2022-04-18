import os
import numpy as np
import json
from PIL import Image

def compute_convolution(I, T, stride=None,padding=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    
    # I assume that the template has the template is also indexed by channels.
    (n_rows,n_cols,n_channels) = np.shape(I)
    (t_rows, t_cols, t_channels) = np.shape(T)

    '''
    BEGIN YOUR CODE.
    '''
    if padding == None:
        heatmap = np.zeros((n_rows, n_cols))
        it = np.nditer(heatmap, flags=['multi_index'], op_flags=['writeonly'], order='C')
        for x in it:
            sum = 0.0
            norm = 0.0
            i = it.multi_index[0]
            j = it.multi_index[1]
            if i >= n_rows - t_rows + 1 or j >= n_cols - t_cols + 1:
                continue;
            t_it = np.nditer(T, flags=['multi_index'], order='C')
            for t in t_it:
                t_i = t_it.multi_index[0]
                t_j = t_it.multi_index[1]
                channel = t_it.multi_index[2]
                pixel = I[i+t_i][j+t_j][channel]
                norm += pixel*pixel
                sum += pixel*t
            x[...] = sum/np.sqrt(norm)

    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap, template_shape):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    box_height = template_shape[0]
    box_width = template_shape[1]
    
    threshold = 0.5
    it = np.nditer(heatmap, flags=['multi_index'])
    for x in it:
        if x > threshold:
            tl_row = it.multi_index[0]
            tl_col = it.multi_index[1]
            br_row = tl_row + box_height
            br_col = tl_col + box_width           
            output.append([tl_row,tl_col,br_row,br_col, x])
        

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    template_path = './data/teamplate/template.jpg'
    T = Image.open(template_path)
    T = np.asarray(T)

    heatmap = compute_convolution(I, T)
    output = predict_boxes(heatmap, np.shape(T))

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = './data/RedLights2011_Medium'

# load splits: 
split_path = './data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = './data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)

import streamlit as st

import cv2
from PIL import Image, ImageDraw, ImageFont
import imageio


import tempfile			
import os
from time import sleep
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
# from yolo_utils import (read_classes, read_anchors, generate_colors, preprocess_image, 
# draw_boxes, yolo_eval)

import colorsys
import imghdr
import random

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Step 1: Compute box scores
    ### START CODE HERE ### (≈ 1 line)
    box_scores = box_confidence * box_class_probs
    ### END CODE HERE ###
    
    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    ### START CODE HERE ### (≈ 2 lines)
    box_classes = K.argmax(box_scores, axis = -1)
    box_class_scores = K.max(box_scores, axis = -1, keepdims = None)
    ### END CODE HERE ###
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    ### START CODE HERE ### (≈ 1 line)
    filtering_mask = box_class_scores >= threshold
    ### END CODE HERE ###
    
    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    ### END CODE HERE ###
    
    return scores, boxes, classes

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """

    # Assign variable names to coordinates for clarity
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    
    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = max(xi2 - xi1, 0)
    inter_height = max(yi2 - yi1, 0)
    inter_area = inter_width * inter_height

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1_y2 - box1_y1) * (box1_x2 - box1_x1)
    box2_area = (box2_y2 - box2_y1) * (box2_x2 - box2_x1)
    union_area = (box1_area + box2_area) - inter_area
    
    # compute the IoU
    iou = inter_area / union_area
    
    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    # Given the boxes kept and it's scores, it will iterate to leave the one with the highest prob.
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=max_boxes, iou_threshold=iou_threshold)

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    # Using the index for the boxes left, get the scores, box, and classes for each.
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """

    # Retrieve outputs of the YOLO model
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes

def yolo_head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.
    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.
    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # Static implementation for fixed models.
    # TODO: Remove or add option for static implementation.
    # _, conv_height, conv_width, _ = K.int_shape(feats)
    # conv_dims = K.variable([conv_width, conv_height])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    # Static generation of conv_index:
    # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
    # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
    # conv_index = K.variable(
    #     conv_index.reshape(1, conv_height, conv_width, 1, 2))
    # feats = Reshape(
    #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs


def clearDir(path):
	for filename in os.listdir(path):
	    file_path = os.path.join(path, filename)
	    try:
	        if os.path.isfile(file_path) or os.path.islink(file_path):
	            os.unlink(file_path)
	        
	    except Exception as e:
	        print('Failed to delete %s. Reason: %s' % (file_path, e))


def getMiddleFrame(videofile):
	vidcap = cv2.VideoCapture(tfile.name)
	frames_no = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/2)
	vidcap.set(cv2.CAP_PROP_POS_FRAMES, frames_no)
	_, frame_middle = vidcap.read()
	return frame_middle

def lineConfig(img,positionx, ylower=-1, yupper=-1):
    shape_img = img.shape
    if ylower == -1:
        ylower = 0
    if yupper == -1:
        yupper = shape_img[0]
    if positionx >= shape_img[1]:
        cv2.line(img, (ylower,shape_img[1]),(yupper,shape_img[1]),(255,127,0), 3)
    elif positionx <= 0:
        cv2.line(img, (ylower,0),(yupper,0),(255,127,0), 3)
    else:
        cv2.line(img, (ylower,positionx),(yupper, positionx),(255,127,0), 3)
    return img

def getVehicles(video, positionx,
               xlow,xhigh, offset, lmin):
    
    length_min=lmin #minimum length of boxes
    height_min=lmin #minimum height of boxes

    delay= 60 #FPS

    vehicles = []
    bounds = []
    cars= 0


    xlow = xlow #0
    xhigh = xhigh #frame.shape[1]
    offset= offset #error permited per pixel

    cap = cv2.VideoCapture(video)
    subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

    
    count=0
    while True:
        ret , frame2 = cap.read()
        if ret:
            tempo = float(1/delay)
            sleep(tempo) 
            grey = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grey,(3,3),5)
            img_sub = subtractor.apply(blur)
            dilat = cv2.dilate(img_sub,np.ones((5,5)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dillated = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
            dillated = cv2.morphologyEx (dillated, cv2. MORPH_CLOSE , kernel)
            contours,h=cv2.findContours(dillated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            frame3 = frame2.copy()
            cv2.line(frame2, (xlow, positionx), (xhigh, positionx), (255,127,0), 3) 
            for(_,c) in enumerate(contours):
                (x,y,w,h) = cv2.boundingRect(c)
                valid_contours = (w >= length_min) and (h >= height_min)
                if not valid_contours:
                    continue

                cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,0),2)        
                center = getCentroid(x, y, w, h)
                vehicles.append(center)
                bounds.append((x,y,x+w,y+h))
                cv2.circle(frame2, center, 4, (0, 0,255), -1)

                for (x,y) in vehicles:
                    if y<(positionx+offset) and y>(positionx-offset) and x>=xlow and x<=xhigh:
                        cars+=1
                        i = vehicles.index((x,y))
                        box= bounds[i]
                        x1 = box[1]-10
                        if x1 <0:
                            x1 = 0
                        x2 = box[3] + 10
                        if x2 > frame.shape[0]:
                            x2 = frame.shape[0]
                        y1 = box[0] - 10
                        if y1 <0:
                            y1 = 0
                        y2 = box[2] + 10
                        if y2 > frame.shape[1]:
                            y2 = frame.shape[1]
                        veh.image(frame3[x1:x2, y1:y2], use_column_width=True, channels ='BGR')
                        cv2.imwrite('output/vehicles/' + "\\frame%d.jpg" % count, frame3[x1:x2, y1:y2])
                        count = count + 1
                        cv2.line(frame2, (xlow, positionx), (xhigh, positionx), (0,127,255), 3) 
                        vehicles.remove((x,y))
                        bounds.pop(i)

            cv2.putText(frame2, "Vehicle Count : "+str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
            vid.image(frame2, use_column_width=True, channels ='BGR')
            dil.image(dillated, use_column_width=True)

            if stop_btn:
            	break
        else:

            break
    cap.release()


        
def getCentroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

def transformPredict(lanes, core_threshold):
    images = [img for img in os.listdir("output/vehicles/") if img.endswith(".jpg")]
    images = pd.Series(images).str.replace('frame','').str.replace('.jpg', '').astype(int)
    images = ('frame' + pd.Series(np.sort(images)).astype(str) + '.jpg').tolist()


    counter = []
    pro = 0
    step = 1/len(images)
    for img in images:
        transformImage('output/vehicles/'+img)
        _, _, out_classes = predict(sess, 'output/transformed/'+img[:-4] + '_transformed.jpg', lanes,core_threshold,0.01)
        pro += step
        progress_bar.progress(pro)
        stat.text("{}% Complete".format(pro*100))
        counter.append(out_classes)
        
    return counter

def predict(sess, img, max_boxes = 1, score_threshold=.01, iou_threshold=.01):
    # Load image size
    input_image = imageio.imread(img)
    shape = (float(input_image.shape[0]), float(input_image.shape[1]))
    
    ## outputs
    scores, boxes, classes = yolo_eval(yolo_outputs, shape, max_boxes=max_boxes,
                                       score_threshold=score_threshold,iou_threshold=iou_threshold)

    # Preprocess your image
    image, image_data = preprocess_image(img, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    ### START CODE HERE ### (≈ 1 line)
    out_scores, out_boxes, out_classes = sess.run(fetches=[scores, boxes, classes],
                                                 feed_dict={yolo_model.input: image_data,
                                                           K.learning_phase(): 0})
    # Filter by desirerd classes
    indexes = []
    index = 0
    for cl in [class_names[i] for i in out_classes]:
        if cl in [class_names[i] for i in [1,2,3,5,7]]:
            indexes.append(index)
        index += 1
        
    out_scores = np.array([out_scores[i] for i in indexes])
    out_boxes = np.array([out_boxes[i] for i in indexes])
    out_classes = np.array([out_classes[i] for i in indexes])


    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save('output/predicted/'+ img[19:-4]+ '_predicted.png', quality=90)
    # Display the results in the notebook
    output_image = Image.open('output/predicted/'+ img[19:-4]+ '_predicted.png')
    veh_class.image(output_image, use_column_width=True)

    
    return out_scores, out_boxes, out_classes

def transformImage(path, min_size=608):
    im = Image.open(path)
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), (0, 0, 0))
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    new_im.save('output/transformed/'+path[16:-4] + '_transformed.jpg')


def finaliseCounter(counter):
    final_counter = {
        'car':0,
        'motorbike':0,
        'truck':0,
        'bus':0,
        'bicycle':0
    }
    for c in counter:
        for i in range(len(c)):
            res = class_names[c[i]]
            final_counter[res]+=1
                
    return final_counter




st.sidebar.title('Navigation')
tab = st.sidebar.radio('Go to', 
	['Instructions', 'Main'],
	 index = 0)
st.title('Car Counter')

if tab == 'Instructions':
	st.header('Steps:')
	st.markdown('**1. Upload mp4 video**')
	st.image('screenshots/1.jpg', use_column_width=True)
	st.markdown("After you upload your video it should appear below, along with a frame of the video.")
	st.markdown('**2. Configure line**')
	st.image('screenshots/2.jpg', use_column_width=True)
	st.markdown("Now you need to configure the line's position. This line will be used to capture all vehicles passing it. 'y' is the y coordinate of the line and 'x0','x1' are the lower and upper x coordinates of the line respectively.")
	st.markdown('**3. Configure vehicle boxes**')
	st.image('screenshots/3.jpg', use_column_width=True)
	st.markdown("""
		Now it's time to set up some parameters for the vehicles's boxes. 
		Specifically the minimum height and width of the boxes as well as the offset (per pixel error) in the line. 
		Then hit the button to start the video capture. This is a trial and error process. 
		The way of thinking is the following. If you capture the same vehicle twice or even more times, 
		when it passes the line, decrease the offset, 
		if you dont capture some vehicles passing by, increase the offset. If the boxes of the 
		cars don't show up in the video
		decrease the minimum length/height, if there are small boxes inside the vehicles increase it.""")
	st.markdown('**4. Capture the Vehicles**')
	st.image('screenshots/4.jpg', use_column_width=True)
	st.markdown("""
		When everything is set up and you've started the video capturing process the vehicles captures should start 
		showing up in the bottom. The process will end automatically when the video ends, but 
		if you want to stop earlier for whatever reason press the 'Stop Capture' button.
		""")
	st.markdown('**5. Configure predictive parameters**')
	st.image('screenshots/5.jpg', use_column_width=True)
	st.markdown("""
		When the video finishes the number of vehicles captured will show up if you are satisfied 
		with the results then you should set up the final parameters. The number of lanes in the road
		captured and the minimum confidence for the model to classify them. As a rule of thumb 0.2
		is a good value for the core probability and it is good to be between 0.1 and 0.5. If your vehicles
		aren't classified at all try decreasing it and if there are multiple wrong classifications try increasing it.
		""")
	st.markdown('**6. Classify the Vehicles**')
	st.image('screenshots/6.jpg', use_column_width=True)
	st.markdown("""
		When both parameters are set up start classifying them. They should start coming up with a 
		bounding box around them.
		""")
	st.markdown('**7. Results**')
	st.image('screenshots/7.jpg', use_column_width=True)
	st.markdown("""
		After the classification is finished, a dictionary will come up containing the number
		of each type of vehicle in the video
		""")
	st.markdown('For further information visit the original repository for this project: [click here](https://github.com/GeorgeEfstathiadis/Vehicle_Counter)')
	st.sidebar.info('Select Main to proceed to the app')

elif tab == 'Main':
	f = st.file_uploader("Upload file", type = ['mp4']) 
	if f is not None:	
		f_size = round(f.size/(10**6),2)
		f2 = f.read()
		st.video(f2)
		st.text('File Name: ' + f.name + '\nFile Type: ' + f.type + '\nFile Size: ' + str(f_size) + 'mb')

		st.header('Line Configuration')
		tfile = tempfile.NamedTemporaryFile(delete=False)
		tfile.write(f2)
		frame = getMiddleFrame(tfile.name)
		stframe = st.empty()

		st.sidebar.header('Line Configuration')
		y = st.sidebar.slider('Position y', 0, frame.shape[0], 0)
		x0 = st.sidebar.slider('Lower x', 0, frame.shape[1], 0)
		x1 = st.sidebar.slider('Upper x', 0, frame.shape[1], 0)

		if x0 > x1:
			st.error('Upper x needs to be larger than lower x.')
		else:
			frame1 = lineConfig(frame, y, x0, x1)
			stframe.image(frame1, use_column_width=True, channels ='BGR')

			if (y!=0) and (x0!=0 or x1!=0):
				st.sidebar.header('Vehicle Capture')
				offset = st.sidebar.slider('Offset', 0, 50, int(frame.shape[0]/120))
				lhmin = st.sidebar.slider('Minimum Length/Height (vehicle box)', 0, int(frame.shape[0]/5), int(frame.shape[0]/10))

				

				st.header('Vehicle Capture')
				if st.button('Start Video Capture'):
					clearDir('output/vehicles')

					vid = st.empty()
					dil = st.empty()
					veh = st.empty()
					stop_btn = st.button('Stop Capture')

					getVehicles(tfile.name, y,x0,x1, offset, lhmin)
				
				st.write('Vehicles captured: ', len(os.listdir('output/vehicles')))
				if len(os.listdir('output/vehicles'))>0:
					st.sidebar.header('Classification')
					lanes = st.sidebar.slider('Lanes', 1, 6, 1)
					core_threshold = st.sidebar.slider('Prediction Threshold', 0., 1., 0.2, step = 0.01)

					st.header('Classification')

					if st.button('Start Classification'):
						clearDir('output/transformed')
						clearDir('output/predicted')

						## Model
						sess = K.get_session()
						class_names = read_classes("model/coco_classes.txt")
						anchors = read_anchors("model/yolo_anchors.txt")
						yolo_model = load_model("model/yolo.h5")
						yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
						yolo_outputs = (yolo_outputs[2],yolo_outputs[0], yolo_outputs[1], yolo_outputs[3])

						veh_class = st.empty()
						progress_bar = st.progress(0)
						stat = st.empty()
                        
						counter = transformPredict(lanes, core_threshold)
						final_counter = finaliseCounter(counter)

						progress_bar.empty()
						veh_class.empty()

						st.header('Results')
						st.json(final_counter)


	else:
	    st.error('File not uploaded.')




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

from help_funcs import (read_classes, read_anchors, generate_colors, preprocess_image, 
draw_boxes, yolo_eval, yolo_head)

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




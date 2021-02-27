# Vehicle_Counter

 A vehicle counter that accepts a video of a road, counts the vehicles passing by and classifies them using OpenCV and Tensorflow.

# Instruction for use

**1. Upload mp4 video**  
![](screenshots/1.jpg)  
After you upload your video it should appear below, along with the first frame.  

**2. Configure line**  
![](screenshots/2.jpg)   

Now you need to configure the lines position. This line will be used to capture all vehicles passing it. y is the y coordinate of the line and x0,x1 are the lower and upper x coordinates of the line respectively.  

**3. Configure vehicle boxes**    
![](screenshots/3.jpg)  

Now its time to set up some parameters for the vehicless boxes.   
Specifically the minimum height and width of the boxes as well as the offset (per pixel error in the line. 
Then hit the button to start the video capture. This is a trial and error process. 
The way of thinking is the following. If you capture the same vehicle twice or even more times,  
when it passes the line, decrease the offset,   
if you dont capture some vehicles passing by, increase the offset. If the boxes of the   
cars dont show up in the video  
decrease the minimum length/height, if there are small boxes inside the vehicles increase it.

**4. Capture the Vehicles**  
![](screenshots/4.jpg)    

When everything is set up and youve started the video capturing process the vehicles captures should start 
showing up in the bottom. The process will end automatically when the video ends, but 
if you want to stop earlier for whatever reason press the Stop Capture button.  
  
**5. Confugure predictive parameters**  
![](screenshots/5.jpg)  

When the video finishes the number of vehicles captured will show up if you are satisfied 
with the results then you should set up the final parameters. The number of lanes in the road
captured and the minimum confidence for the model to classify them. As a rule of thumb 0.2
is a good value for the core probability and it is good to be between 0.1 and 0.5. If your vehicles
arent classified at all try decreasing it and if there are multiple wrong classifications try increasing it.  

**6. Classify the Vehicles**  
![](screenshots/6.jpg)  

When both parameters are set up start classifying them. They should start coming up with a   
bounding box around them.  
	
**7. Results**  
![](screenshots/7.jpg)  

After the classification is finished, a dictionary will come up containing the number
of each type of vehicle in the video.  
	

# PublicCam-Car-Recognition

# Description
This is a python program that uses 2 models separately on a public webcam, to identify cars moving by. It uses OpenCV to open up and create the connection to the Public WebCam, and MobileNETSSD Model for Car recognitioning for the first model, and YoloV3 as the second model.

# How to install and run the application

In order to run the application, you need to download the zipfile from this github repo, install the necessary packages ( numpy and opencv).

So, step by step, it would be : 

1. Download the zipfile
2. Open it using a code editor ( i'm going to show it using PyCharm ) 
3. Install the needed packages by going into the terminal, and typing : 

![image](https://user-images.githubusercontent.com/93039914/232247811-0fdbc6a7-b015-4dc6-992a-9fbdad86116c.png)
![image](https://user-images.githubusercontent.com/93039914/232247838-bbdd8d25-7957-4614-ad72-8d4523986d72.png)

4. After successfully installing the needed packages, now you can press the run button in the top right corner  : ![image](https://user-images.githubusercontent.com/93039914/232247867-0e6ddba6-ea5f-4e6b-956b-bb9c79d6f3db.png)

5. This is how it should look : 
![Test12312313](https://user-images.githubusercontent.com/93039914/232247981-5278af24-af0c-4268-8031-d3574908239c.jpg)

6. If you want to stop the program, you can either press the "q" button , or press the red square inside the program : ![image](https://user-images.githubusercontent.com/93039914/232248011-99baf9c4-dad2-49aa-84b8-d35b56ddf8e4.png)

# How does the algorithm work 
This program uses the YoloV3 model and MobileNetSSD. They are using convulation neural networks ( CNN's ) , but they have a different processing process so to say. 
We have the config and weights needed for the models, and after that we run a for loop going through the detections and if they're score is good enough we are drawing a box around the object and text, indicating the object detected. I've set the conf threshold 0.5 by default. 

# MobileNetSSD 
MobileNetSSD Algorithm : ![image](https://user-images.githubusercontent.com/93039914/232248149-973d3329-ec67-4f99-beda-bf4f339deff6.png)

Here, after we read each frame, in the while true loop, using the blob ( that is an array of image data ), we resize it and normalize it, ( for example the 300 by 300 is the resize value, the 0.007843 is equivalent to dividing each pixel by 255 , that resulting in every pixel being in the range of 0 to 1, so we pretty much take care of the lighting of the image so the model can process it without problems.







# Water-Meters

The aim of this project was to develop a program that provides meter readings from photos. As part of the pipeline, I carried out various preprocessing steps on the images. I then put the images through a UNet model with their masks; the goal was to obtain mask predictions for reading locations on meters. With the predictions masks, I cropped the images of meters to extract only the reading portion. I then manually placed bounding boxes and labels around each digit in 200+ cropped meters. Using the segmented images, I trained a Faster RCNN model to recognize digits. Next, I parsed the predictions from the Faster RCNN model and organized the labels (digits) based on the x-coordinate from each bounding box from left to right. This provided me with the detected digits in proper order. Based on the types of meters included in the training set, I set the decimal point based on the number of digits recognized. On the testing dataset, the Faster RCNN model's mAP was around 96%.

Once the models were trained and preprocessing steps outlined, I created an application that allows a user to upload and image and obtain a reading. This application was dockerized and tested locally, and was found to function well. I tried pushing the application to Heroku, but it consumes too much memory for it to run without a decent subscription. Nonetheless, that the container functions as expected demonstrates that one could productize this application if one has an appropriate server.

# Use an official Python runtime as a parent image
FROM pleasework:latest

COPY . /app
WORKDIR /app

RUN pip install opencv-python
RUN pip install tensorflow
RUN pip install pycocotools
RUN pip install -r requirements.txt
RUN pip install segmentation_models

#ENV NAME World
CMD ["python", "app.py"]

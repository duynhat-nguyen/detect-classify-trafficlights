# Traffic lights detection and classification
The aim of this repository is to detect and classify traffic lights signals (only red and green). This is the final course project of Pattern Recognition course at [University of Information Technology](uit.edu.vn).

1. Dataset  
- At the time of this project, I could not use [this tool](https://github.com/hardikvasa/google-images-download) to crawl images from Google Image, then I took images from [BDD100K dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd/).  
- I reused [this notebook](https://github.com/shirokunet/lane_segmentation/blob/master/tool/04-03_generate_highway_dataset_json.ipynb) to create [one for my need](https://github.com/nhat-nguyenduy/traffic-lights-detection-classification/blob/master/04-03_generate_highway_dataset_json.ipynb) which will only take images in [daytime, night], [highway, city street] scenes, [clear, rainy] weather.
- After that, I used [labelImg](https://github.com/tzutalin/labelImg) to label my images with 2 class green and red (Pascal VOC format). Because my dataset has 80 images (<1000) so I can use [Roboflow](https://roboflow.ai/) to create the TFRecord file from the images and labels.

2. Install  
- I used Tensorflow Object Detection API which is install in a [Colab notebook](https://colab.research.google.com/drive/10wn1XnTjOgupefn-csrjH7KZwYk79bqQ?usp=sharing). More information of how to used a pre-trained model from Tensorflow Model Zoo can be found [here](https://github.com/tensorflow/models/tree/master/research/object_detection). 

3. Models  
- I first used SSD Inception V2. 

4. GUI app  
My [GUI app](https://github.com/nhat-nguyenduy/traffic-lights-detection-classification/blob/master/GUI.py) was created based [an app](https://github.com/streamlit/demo-self-driving) using [Streamlit library](https://www.streamlit.io/)

5. How to run  

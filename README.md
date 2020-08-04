# Traffic lights detection
The aim of this repository is to detect traffic lights (only red and green). This is the final course project of Pattern Recognition course at [University of Information Technology](uit.edu.vn).
```
.
├── data
│   ├── raw
│   │   └── BDD100k
│   └── tfrecord
│       ├── README.roboflow.txt
│       ├── train
│       └── valid
├── LICENSE
├── models
│   ├── pretrained
│   │   └── ssd_inception_v2_coco_2018_01_28
│   └── training
│       └── ssd_inception_v2_coco
├── README.md
├── requirements.txt
└── src
    ├── api
    │   ├── install_dependencies.sh
    │   ├── install_tfodapi.sh
    │   └── models
    ├── data
    │   ├── manually
    │   └── roboflow
    ├── models
    │   ├── config
    │   ├── pretrained
    │   └── train
    └── visualization
        └── GUI.py
```
1. Dataset  
- At the time of this project, I could not use [this tool](https://github.com/hardikvasa/google-images-download) to crawl images from Google Image, then I took images from [BDD100K dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd/).  
- I reused [this notebook](https://github.com/shirokunet/lane_segmentation/blob/master/tool/04-03_generate_highway_dataset_json.ipynb) to create [one for my need](https://github.com/nhat-nguyenduy/traffic-lights-detection-classification/blob/master/04-03_generate_highway_dataset_json.ipynb) which will only take images in [daytime, night], [highway, city street] scenes, [clear, rainy] weather.
- After that, I used [labelImg](https://github.com/tzutalin/labelImg) to label my images with 2 class green and red (Pascal VOC format). Because my dataset has 80 images (<1000) so I can use [Roboflow](https://roboflow.ai/) to create the TFRecord file from the images and labels.

2. Install  
- I used Tensorflow Object Detection API which is installed in a [Colab notebook](https://colab.research.google.com/drive/10wn1XnTjOgupefn-csrjH7KZwYk79bqQ?usp=sharing). More information of how to use a pre-trained model from Tensorflow Model Zoo can be found [here](https://github.com/tensorflow/models/tree/master/research/object_detection). 

3. Models  
- SSD Inception V2
- SSD MobileNet V2
- SSD ResNet 50 FPN

4. Training  
Use [this notebook](https://colab.research.google.com/drive/10wn1XnTjOgupefn-csrjH7KZwYk79bqQ?usp=sharing) for training. 

5. Inference  
Use [this notebook](https://colab.research.google.com/drive/1vuXUlLNloK0Su6kdDDoDWtgEvsMZdEMV?usp=sharing) for inference. The [GUI app](https://github.com/nhat-nguyenduy/traffic-lights-detection-classification/blob/master/GUI.py) was created based [an app](https://github.com/streamlit/demo-self-driving) using [Streamlit library](https://www.streamlit.io/)

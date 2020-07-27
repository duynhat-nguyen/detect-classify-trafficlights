mkdir -p ./models/pretrained/ssd
mkdir ./models/pretrained/faster_rcnn
cd ./models/pretrained/ssd

# SSD Inception V2
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz
rm ssd_inception_v2_coco_2018_01_28.tar.gz

# SSD MobileNet V2
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz

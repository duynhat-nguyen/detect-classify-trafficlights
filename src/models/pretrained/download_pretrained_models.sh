mkidr -p models/pretrained && cd models/pretrained
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
tar -xvf ./models/pretrained/ssd_inception_v2_coco_2018_01_28.tar.gz
rm ssd_inception_v2_coco_2018_01_28.tar.gz

mkdir ./src/api/tensorflow && cd ./src/api/tensorflow
git clone https://github.com/tensorflow/models.git 
cd models/research
protoc object_detection/protos/*.proto --python_out=.

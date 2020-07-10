cd ./src/api/
git clone https://github.com/tensorflow/models.git 
cd models/research
protoc object_detection/protos/*.proto --python_out=.
python object_detection/builders/model_builder_tf1_test.py

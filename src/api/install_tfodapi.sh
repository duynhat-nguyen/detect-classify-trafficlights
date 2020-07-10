git clone https://github.com/tensorflow/models.git ./src/api/
protoc ./src/api/models/research/object_detection/protos/*.proto --python_out=.
python ./src/api/models/research/object_detection/builders/model_builder_tf1_test.py

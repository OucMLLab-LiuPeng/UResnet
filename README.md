# UResnet
A method to to underwater image enhancement.  
CycleGan is used to generate paired images to train UResnet.
# Samples of Training Set
![trainset](/images/trainingset.jpg)
# Samples of Model Outputs
![results](/images/results.jpg)
# Requirements
- Linux (code have not been tested on Windows or OSX)
- NVIDIA GPU + CUDA CuDNN
- pytorch >= 0.4.1
# Train model
use python train.py --input_images_path xxx/path/to/input/ --label_images_path xxx/path/to/label --snapshots_folder xxx/path/to/save/snapshot  
use python train.py -h to choose more usable options

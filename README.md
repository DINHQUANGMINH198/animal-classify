<!-- install env : -->
str2bool
pandas
<!-- anaconda -->
pip install -U scikit-learn scipy matplotlib
<!-- install pytorch -->
https://pytorch.org/get-started/locally/
# Pytorch Animals
Multi-label image classification with transfer learning.

# Using the Dataset:
1. Classification data
2. Move the extracted ``animals`` folder into ``data/`` directory.

The final directory structure should look like:
# Running Instructions:
Before the first run, prepare the image folders using:
1 animal - 1 folder
<!-- final data -->
chicken    -> Divided into 696 train / 175 val / 218 test   -> Copied!
deer       -> Divided into 788 train / 197 val / 247 test   -> Copied!
monkey     -> Divided into 1351 train / 338 val / 423 test   -> Copied!
rabbit     -> Divided into 689 train / 173 val / 216 test   -> Copied!
tiger      -> Divided into 1239 train / 310 val / 388 test   -> Copied!
<!-- Run -->
<!-- split data train : -->
python prepare.py
<!-- train -->

<!-- PyTorch Animals

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED
  --batch_size BATCH_SIZE
  --epochs EPOCHS
  --lr LR
  --weight_decay WEIGHT_DECAY
  --optimizer OPTIMIZER
                        sgdm or adam
  --input_size INPUT_SIZE
  --display_images DISPLAY_IMAGES
  --pretrained PRETRAINED
  --save SAVE
  --device DEVICE       cpu or cuda
  --checkpoint CHECKPOINT
                        path checkpoint -->


python main.py --epoch=200
<!-- resume train cpu with checkpoint-->
python main.py --epoch=200 --checkpoint models/pt_20221129_220703_1.000000.pth



To plot results:

```bash
python plot.py --json_path='dumps/xxx/args.json'
```

To predict an image using a checkpoint:
<!-- Test -->
```bash
python predict.py --model_path models/pt_20221201_002531_0.997485.pth --img_path SplitCaptcha/SplitCaptcha/0a1ac6b5.jpg
```
<!-- Using model checkpoint to classify all data -->
python classify_data.py
# References:
* Pre-trained models: https://pytorch.org/docs/master/torchvision/models.html
* Tutorial 1: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
* Tutorial 2: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
* Tutorial 3: https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce
* YouTube Playlist: https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
* Data Loader Example: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py
* Optimizer state-dict: https://discuss.pytorch.org/t/importance-of-optimizers-when-continuing-training/64788

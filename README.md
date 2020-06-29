## SENet-50 for feature extraction

#### Architectire

SE-ResNet-50 trained with standard softmax loss on MS-Celeb-1M and fine-tuned on VGGFace2

#### Installation

Organise the dataset directory as follows:

``` Shell
./
  resources/
    casia_boxes_refined.csv
  images/
    casia/
       ...
  model/
    senet50_256.pth
```

Use the [following link](https://drive.google.com/drive/folders/1-CHt4UWZRNagvPZzd-h06cLt0uktC-hw?usp=sharing) to download all data necessary for feature extraction. You should also place CASIA dataset to ```casia``` folder.

#### Extraction

To exctract feature vectors use the follwing script

``` Shell
python extraction.py --dataset casia --bb_file casia_boxes_refined.csv
```
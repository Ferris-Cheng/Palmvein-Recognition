# Palmvein Recognition using Tensorflow
>This is a Tensorflow implementation of the palmvein recognizer. The project also uses ideas from the paper ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832, '').

## Environment
The code is tested using Tensorflow r1.6 under Windows10 with Python 3.5.

## Inspiration

>The code is heavily inspired by the ["FaceNet"](https://github.com/davidsandberg/facenet, '') implementation.

## Contactless Palmvein Image Dataset

The ["Tongji Contactless Palmvein Dataset"](http://sse.tongji.edu.cn/linzhang/contactlesspalmvein/index.htm, '') has been used for train and test. The dataset consists of total of 12000 images over 600 identities after palmvein detection.

## ROI
The ["ROI image"](https://drive.google.com/open?id=1P_AfQNAK36rDzZnBjgxnIAfRMLxYiTg8, '') has been used for experiment.<br />
![Palmvein ROI e.g.](./data/00001.bmp)
&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;
![Palmvein ROI e.g.](./data/00014.bmp)

## Training
We choose the first 360 classes in the palmvein ROI dataset for training. This training set consists of total of 7200 images over 360 identities.

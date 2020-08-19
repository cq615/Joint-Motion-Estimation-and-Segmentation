# Joint Learning of Motion Estimation and Segmentation for Cardiac MR Image Sequences

Code accompanying MICCAI 2018 paper of the same title. Paper link: https://arxiv.org/abs/1806.04066



## Usage

Lasagne and theano implementation of the framework.

main.py ==> main training file

test_prediction.py ==> applies joint prediction and visualisation

models ==> proposed network and layers

dataio ==> includes loading images, data_augmentation, etc

utils ==> metrics and visualisation

model ==> Model parameters

test ==> One test sample

### News: Update Pytorch implementation of the work in pytorch_version.

pytorch_version ==> pytorch implementation of the models

![](joint_model.pdf)

## Citation and Acknowledgement
If you use the code for your work, or if you found the code useful, please cite the following works:

Qin, C., Bai, W., Schlemper, J., Petersen, S.E., Piechnik, S.K., Neubauer, S. and Rueckert, D. Joint learning of motion estimation and segmentation for cardiac MR image sequences. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2018: 472-480.

C. Qin, W. Bai, J. Schlemper, S. Petersen, S. Piechnik, S. Neubauer and D. Rueckert. Joint Motion Estimation and Segmentation from Undersampled Cardiac MR Image. International Workshop on Machine Learning for Medical Image Reconstruction, 2018: 55-63.


## Licence
This project is licensed under the terms of the MIT license.

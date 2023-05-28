# Semi-Supervised Specific Emitter Identification via Dual Consistency Regularization
The code corresponds to the IoTJ paper

# Requirement
pytorch 1.10.2
python 3.6.13

# Abstract
Deep Learning (DL)-based Specific Emitter Identification (SEI) is a potential physical layer authentication technique for Industrial Internet-of-Things (IIoT) Security, which detects the individual emitter according to its unique signal features resulting from transmitter hardware impairments. The success of DL-based SEI often depends on sufficient training samples and the integrity of samples' labels. The extensive deployment of wireless devices generates a huge amount of signals, but signals labeling is quite difficult and expensive with the high demand for expertise. In this paper, we present a SEI method based on Dual Consistency Regularization (DCR), which enables feature extraction and identification using a few labeled samples and a large number of unlabeled samples. With the help of pseudo-labeling, we leverage consistency between the predicted class distribution of weakly-augmented unlabeled training samples and that of strongly-augmented training unlabeled samples, and consistency between semantic feature distribution of labeled samples and that of pseudo-labeled samples, which takes the unlabeled samples into account to model parameter tuning for a more accurate emitter identification. Extensive numerical results demonstrate that compared with well-known semi-supervised learning-based SEI methods, our method obtains 99.77% identification accuracy on a Wi-Fi dataset and 90.10% identification accuracy on an Automatic Dependent Surveillance-Broadcast (ADS-B) dataset when only 10% of training samples are labeled, and improves the identification accuracy on the Wi-Fi dataset and the ADS-B dataset by more than 19.07% and 5.30%, respectively.

# Dataset
We use the dataset proposed in paper [55] and [56] to evaluate our proposed MAT-based SS-SEI method. The former is a large-scale real-world radio signal dataset based on a special aeronautical monitoring system, ADS-B, and the latter is WiFi dataset collected from USRP X310 radios that emit IEEE 802.11a standards compliant frames. The number of categories of ADS-B dataset and WiFi dataset is 10 and 16, respectively. The length of each sample of ADS-B dataset and WiFi dataset is 4,800 and 6,000, respectively. The number of training samples of ADS-B dataset and WiFi datsset is 3, 080. The number of testing samples of ADS-B dataset and WiFi dataset is 1,000 and 16,004, respectively. We construct five semi-supervised scenarios and one fully supervised scenario, where the number of labeled training samples to the number of all training samples ratio is {5%, 10%, 20%, 50%, 100%}, to evaluate the identification performance of the proposed SS-SEI method. In addition, 30% of the training samples is used as the validating samples during the training process.

[55] Y. Tu, Y. Lin, et al., “Large-scale real-world radio signal recognition with deep learning,” Chin. J. Aeronaut., vol. 35, no. 9, pp. 35–48, Sept.
2022.

[56] K. Sankhe, M. Belgiovine, F. Zhou, S. Riyaz, S. Ioannidis, and K. Chowdhury, “ORACLE: Optimized radio classification through convolutional neural networks,” in IEEE Conf. Comput. Commun., Apr.2019, pp. 370-378.

The ADS-B dataset can be downloaded from the Link: https://pan.baidu.com/s/13qW5mnfgUHBvWRid2tY2MA Passwd：eogv

The WiFi dataset can be downloaded from https://pan.baidu.com/s/1aah2mh9sJvtXs6XjKkxLZQ and password is biv8.
In the WiFi dataset, we randomly select 3,080 samples from the original dataset (only 62 step) for simulation, and selection approach can be seen from the code Dataset/SplitData.py.

# Classification Accuracy
 Methods  | ADS-B (5%) | ADS-B (10%) | WiFi (5%) | WiFi (10%)
 ---- | ----- | ------  | ----- | ------  |
 CVNN  | 60.50% |  74.50% | 20.47% |28.64%
 DRCN  | 54.20% | 72.40% | 21.94% | 47.51%
 SSRCNN | 49.30% | 79.30% | 19.33% | 38.09%
 TripleGAN | 45.10% | 61.10% | 27.57% | 37.27%
 SimMIM | 65.90% | 77.90% | 31.71% | 49.59%
 MAT-CL | 70.06% | 83.80% | 27.26% | 80.70%
 MAT-PA | 74.00% | 84.80% | 28.82% | 54.96%
 DCR (Proposed)   | 79.70% | 90.10% | 98.99% | 99.77%


# E-mail
If you have any question, please feel free to contact us by e-mail (1020010415@njupt.edu.cn).

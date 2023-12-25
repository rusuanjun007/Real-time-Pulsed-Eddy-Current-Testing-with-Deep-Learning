# Real-time Pulsed Eddy Current Testing with Deep Learning
This is the source code for the paper Real-time Pulsed Eddy Current Testing with Deep Learning.

Pulsed eddy current (PEC) is a crucial eddy current testing (ECT) technique that is widely used in the non-destructive testing (NDT) industry. Automatic recognition of metal thickness
through PEC has the potential to enhance plant operation safety, such as detecting pipe thinning. In this paper, we present a three-fold contribution. Firstly, we construct a new PEC dataset
based on a custom-designed portable PEC device, encompassing a wide range of features. To better simulate real-world application scenarios, measurements on various lift-off, insulation, position,
and weather jacket conditions are included. Secondly, we utilize a 1D convolutional neural network (CNN) to achieve highly accurate automatic thickness recognition, and the recognition
result is not affected by lift-off and edge effects. Lastly, we deploy a compact and lightweight 1D CNN on the STM32 microcontroller embedded in our device. The network achieves real-time, accurate, and low-latency automatic thickness recognition.
Our study represents a significant advancement towards the development of automated thickness recognition technologies in the NDT industry.

Seven different models are trained and their architectures are listed below.

![微信截图_20230420155334](https://user-images.githubusercontent.com/64902728/233404797-95570299-694b-4cb7-b87a-da88581a9be9.png)

![微信截图_20230420155323](https://user-images.githubusercontent.com/64902728/233404803-b76dae66-b087-4c2c-b8d2-93ee99ca6565.png)

The lightweight CNN can be deployed in STM32 to achieve real-time metal thickness recognition, and the architecture is shown below.

![微信截图_20230420155756](https://user-images.githubusercontent.com/64902728/233405945-3d87c85f-05d4-46b2-a2ed-133bee3d6b4e.png)

Our PEC dataset can be downloaded from Kaggle. https://www.kaggle.com/datasets/rusuanjun/pec-dataset

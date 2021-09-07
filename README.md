# MaskCoverage_backend

In the face of the COVID19 outbreaking, people usually wear face mask for protecting themself. In public space, the more people who dosen't wear face mask will leads to the higher risk of cluster infection.

So we developed this project. In this project our model can automatically detect there are how many people in a given photo, and what is the ratio of people among them wearing face mask properly. It can be used to indicate how dangerous this space would be.

## Demo

![](/media/result/test1.jpg)

As you can see, we correctly detect there are 10 people in this photo, and 8 out of them weared face mask.

## Model

We use FastRCNNPredictor provided by torchvision as our pretrained model, and mask data for kaggle [face-mask-detection](https://www.kaggle.com/andrewmvd/face-mask-detection) dataset


<div align="center">    
 
# Age Classification Using Transfer Learning     

[![Blog](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@thisara.weerakoon2001/age-classification-using-transfer-learning-vgg16-d2f240f67d26)


<!--  
Conference   
-->   
</div>

You can have look into my <a href= "https://medium.com/@thisara.weerakoon2001/age-classification-using-transfer-learning-vgg16-d2f240f67d26">medium post</a> for a comprehensive explanation above the code and theory behind the project
 
## Description   
This project explores the use of deep learning to predict a person's age from facial images. Leveraging the VGG16 model, a convolutional neural network pre-trained on the ImageNet dataset, we apply transfer learning to classify age groups.

- **Motivation:** To gain deeper insights into computer vision and its applications, and to contribute to advancements in the field.
- **Why:** Human age estimation based on facial features demonstrates the remarkable capability of our brains.Translating this human skill to machines using deep learning techniques can unlock numerous applications including 
  - **Personalized Marketing:** Tailoring advertisements based on the predicted age group.
  - **Enhanced Security:** Improved surveillance by recognizing age-specific behaviors.
  - **Medical Applications:** Age estimation for planning treatments and predicting health trends.

- **Problem Solved:** Tackling complex computer vision task of classify persons age from his/her facial image.Applying transfer learning to tackle above problem.
- **What We Learned:**
  - **1:** How to preprocess image data
  - **2:** How to build a convolutional neural network from scratch (it not worked for this task by the way :) )
  - **3:** How to train a pre-trained model using transfer learning techniques.
 
  

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   

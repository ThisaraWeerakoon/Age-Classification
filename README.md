
<div align="center">    
 
# Age Classification Using Transfer Learning     

[![Blog](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@thisara.weerakoon2001/age-classification-using-transfer-learning-vgg16-d2f240f67d26)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
	![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)


<!--  
Conference   
-->   
</div>

![1_d6pG5pm8BKBe5uz16sp4zg-ezgif com-webp-to-jpg-converter](https://github.com/ThisaraWeerakoon/Age-Classification/assets/83450623/e02f049b-806e-41a5-9871-bdab48dfefd2)

You can have a look into my <a href= "https://medium.com/@thisara.weerakoon2001/age-classification-using-transfer-learning-vgg16-d2f240f67d26">medium post</a> for a comprehensive explanation about the code and theory behind the project
 
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

 ## Methodology

1. **Data Visualization:**
   - We used the UTKFace dataset, comprising over 20,000 facial images with annotations of age, gender, and ethnicity.
   - Images were visualized to understand the dataset distribution and the embedded labels.

2. **Data Preprocessing:**
   - Images were resized to 224x224 pixels to match the VGG16 model requirements.
   - Normalization was performed to scale pixel values between 0 and 1.
   - Age labels were extracted and categorized into five age groups: 0–24, 25–49, 50–74, 75–99, and 100–124.

3. **Transfer Learning with VGG16:**
   - The VGG16 model, pre-trained on ImageNet, was used as the base model.
   - The model's layers were frozen, and additional dense layers with dropout and L2 regularization were added.
   - The final output layer was designed to classify images into the five age groups using softmax activation.

4. **Model Training:**
   - The model was compiled with categorical cross-entropy loss and the Adam optimizer.
   - Early stopping and model checkpoint callbacks were employed to monitor validation performance and prevent overfitting.
   - The model was trained on 90% of the data and validated on the remaining 10%.

5. **Model Evaluation:**
   - The model's performance was evaluated on the test set, assessing accuracy and loss.
   - Training and validation loss curves were plotted to visualize the learning process and detect potential overfitting.

6. **Age Prediction:**
   - A function was developed to predict the age group of new images.
   - The function preprocesses the input image, makes predictions using the trained model, and maps the predictions to age groups.

## Visualization of the model used
![0_cV6Ciyjm0pdebW_2-ezgif com-webp-to-jpg-converter](https://github.com/ThisaraWeerakoon/Age-Classification/assets/83450623/e3bf3776-6907-4240-987d-5707abcb6ee9)

## Code Implementation

The project's code is organized in a Jupyter notebook, which includes detailed steps for data preprocessing, model training, and evaluation. Key libraries used in the project include:

- `numpy` for numerical operations
- `matplotlib` for data visualization
- `cv2` (OpenCV) for image processing
- `keras` for building and training the neural network
- `visualkeras` for visualizing the model architecture

## Example Usage

To test the trained model on new images, follow these steps:

1. **Preprocess the Image:**
   ```python
   def image_preprocessing(img_path):
       img = cv2.imread(img_path)
       resized_img = cv2.resize(img, (224, 224))
       normalized_img = resized_img / 255.0
       return normalized_img
2.**Predict Age Group:**
 ```python
  def predict_on_image(img_path):
    preprocessed_img = image_preprocessing(img_path)
    reshaped_img = np.reshape(preprocessed_img, (1, 224, 224, 3))
    predicted_labels_probabilities = model.predict(reshaped_img)
    class_index = np.argmax(predicted_labels_probabilities)
    age_class = str(class_index * 25) + "-" + str((class_index + 1) * 25 - 1)
    return age_class
```

3.**Visualize Prediction:**
 ```python
  new_sample_img_rgb = cv2.cvtColor(new_sample_img_bgr, cv2.COLOR_BGR2RGB)
  cv2.putText(new_sample_img_rgb, predicted_age_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  plt.imshow(new_sample_img_rgb)
```

## Credits

We used several third-party assets and tutorials, including:

- [Tensorflow](https://www.tensorflow.org/api_docs)
- [VGG16 Model](https://keras.io/api/applications/vgg/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Badges

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
	![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## How to Contribute

We welcome contributions from the community! If you are interested in contributing, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
    ```sh
    git checkout -b feature-or-bugfix-name
    ```
3. Commit your changes:
    ```sh
    git commit -m "Description of the feature or bug fix"
    ```
4. Push to the branch:
    ```sh
    git push origin feature-or-bugfix-name
    ```
5. Open a pull request and provide a detailed description of your changes.

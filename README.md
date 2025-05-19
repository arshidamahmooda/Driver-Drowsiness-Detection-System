## Driver Drowsiness Detection Using Deep Learning 
# Aim
 This project aims to investigate the effectiveness of combining face detection and eye region analysis for drowsiness detection. A deep learning model, EfficientNetB0, trained on a dataset of labeled images, will classify eye states (open/closed) and yawning (yes/no). The combined predictions from these classifications will be used to provide a robust drowsiness assessment.
 
# Methodology
 
  Dataset Preparation: A dataset of facial images categorised into four classes—Open, Closed, yawn, and no_yawn—was used for training and evaluation. The dataset was split into training 80%, validation 20%, and test sets. The distribution of classes within these sets was analysed to ensure a balanced representation.
  Data Augmentation: To enhance model robustness and prevent overfitting, data augmentation was performed on the training set using ImageDataGenerator. This included random rotations,shifts, shears, zooms, horizontal flips, and brightness adjustments.
  Model Development: A pre-trained EfficientNetB0 model, initially trained on the ImageNet dataset, was used as the base model. Transfer learning was employed to leverage the pre-trained weights for feature extraction. The final layers of EfficientNetB0 were unfrozen, and a custom classification head, consisting of dense layers with dropout and L2 regularisation, was added to adapt the model for the four-class drowsiness detection task.
 Eye Region Cropping and Quality Assessment: To improve the accuracy of drowsiness detection, an algorithm was developed to automatically detect and crop the eye region from input images. This algorithm uses Haar cascades for initial eye detection and a quality assessment function to select the bestcropped eye region based on sharpness, contrast, and completeness.
  Model Training and Validation: The model was trained using the augmented training data and validated on a separate validation set. The Adam optimiser and categorical cross-entropy loss function with label smoothing were used for optimisation. Callbacks, including early stopping, ReduceLROnPlateau, and ModelCheckpoint, were implemented to monitor performance and prevent 
overfitting. The model was trained with optimized hyperparameters for the best results with minimum loss and maximum accuracy.
  Model Evaluation: The trained model was evaluated on the held-out test set using metrics such as accuracy, precision, recall, and the F1-score to assess its performance on unseen data. A confusion matrix was generated to visualize the model's classification performance for each class. The combined predictions for both full-face and cropped eye regions were used for the 
drowsiness state assessment.
  Prediction and Interpretation: For predictions, both the original image and the cropped eye region are preprocessed and fed to the trained model. The model outputs probabilities for each class, and these probabilities are combined to determine a combined confidence score. Based on the predicted class and its confidence, a final determination of "Drowsy" or "Non-Drowsy" is made.
 
 # Result
 
 The model achieved an overall accuracy of 100% on the test set. The overall 
precision and recall were 100% and 100%, respectively. The F1-score, which 
balances precision and recall, was calculated to be 1.00.



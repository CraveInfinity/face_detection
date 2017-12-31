# face_detection

Accuracy : 67% , libraries used : pytorch, scikit-learn, opencv

## Approach:
  1. Creating pairs of images, positive samples - image pairs of same persons
                               negative samples - image pairs of different persons
  2. Care was taken to randomize the selection and ratio of positive to negative is 1:2
  3. Negative samples contains one image of the positive sample pair to give signal boost to the model
  4. Features are created using Convolutional Neural Networks for both the images in a sample and are then concatenated to pass to fully connected layers.
  5. Leakyrelu is used as activation function to avoid any flattening the learning

Documentation:

## preprocessing.py
    Creates combination of positive and negative examples from the given raw image dataset
## model.py
    Pytorch model to extract the features in both the images and construct a image similarity model
## build.py
    training the images on the model and optimize the loss function
## predict.py (under progress)
    sample image and showing the result of the prediction
    
    
## Possible Improvements:
    1. Better loss function could have been used and could be customized to the problem
    2. Logging could be implemented more throughly
    3. Earlystopping and better checkpoint mechanism could be added
    4. More convolutional layers and dense layers could be added, if more gpu could be used
    5. Fine tuning of the hyperparameters, better learning rate 
    6. Better optimizer can be used to compensate the momentum and learning rates.
    7. Further more data augmentations could have been added like contrast, brightness. etc
    
    

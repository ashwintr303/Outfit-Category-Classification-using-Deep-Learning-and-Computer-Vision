# Outfit-Category-Classification-using-Deep-Learning-and-Computer-Vision
Categorization of items into compatible outfits using a CNN model and a finetuned MobileNet model using tf.keras - EE599 CV project

## Dataset description
Polyvore Outﬁts is a real-world dataset created based on users’ preferences of outﬁt conﬁgurations on an online website named polyvore.com: items within the outﬁts that receive high-ratings are considered compatible and vice versa. It contains a total of 365,054 items and 68,306 outﬁts. The maximum number of items per outﬁt is 19. A visualization of an outﬁt is shown in Figure 1.

![Figure 1: A visualization of a partial outﬁt in the dataset. The number at the bottom of each image is the ID of this item.](readme_images/image1.jpg)  

## File description  
1. dataloader.py: Data preprocessing and loading.  
2. model.py: CNN model   
3. model-finetuned-mobilenet: model built using pretrained mobilenet  
4. utils.py: hyperparameters and file paths  

## Dependencies
1. Python 3.7  
2. tensorflow 2.1  
3. tensorflow-gpu 2.1  
4. tqdm 4.43  
5. torchvision 0.5  
6. pillow 7.0.0  
To install the complete list of dependencies, run:  
```
pip install -r requirements.txt
```

## Running the code:  
Set the parameters in utils.py. The code uses CUDA which needs an Nvidia GPU. If not using a GPU, set use_cuda flag to False in utils.py.

## References  
[1] https://github.com/davidsonic/EE599-CV-Project  
[2] https://www.tensorflow.org/api_docs/python/tf/keras/Model

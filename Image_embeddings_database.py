#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def image_embeddings_database(resnet, model):
    """
    Given the images features vectors(Resnet) and trained model, outputs a dictionary 
    that maps Image ID to semantic embeddings with trained model
    
    Parameters
    ----------
    resnet: dict{id - (int), feature vector - np.array - (1,512)}
        Contains all images features vectors
        
    model: the trained Img2Caption
    
    Returns
    -------
    image_database: dict{id - (int), semantic embeddings - np.array - (1,50)}
        A dictionary that maps image feature vectors to semantic embeddings
        
    """
    
    image_database = {}
    
    for idx, features in resnet.items():
        image_database[idx] = model(features)
    
    return image_database


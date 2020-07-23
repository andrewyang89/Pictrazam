
import json
import numpy as np
from collections import defaultdict


# missing embedding function

class Coco:
    def __init__(self):
        """
        Contains dictionaries that convert between properties of the dataset.
        
        Dictionaries:
        ------------
        imageID_to_captions: input image ID and get all associated captions
        
        imageID_to_url: input image ID and output URL
        
        imageID_to_captionID: input image ID to output list of associated caption IDs
        
        captionID_to_caption: input caption ID to ouput associated caption
        
        captionID_to_embedding: input caption ID to ouput associated embedding
        
        
        Properties:
        -----------
        image_ids: all image IDs
        
        caption_ids: all caption IDs
        
        captions: all captions
        
        urls: list of all URLs
        
        """
        
        with open('captions_train2014.json') as json_file:
            data = json.load(json_file)
            
        self.imageID_to_captions = defaultdict(list)
        self.imageID_to_captionID = defaultdict(list)
        
        # iterating through images
        
        self.urls = []
        self.image_ids = []  
        self.imageID_to_url = {}
        
        for i in range(len(data["images"])):
            self.urls.append(data["images"][i]["coco_url"])
            self.image_ids.append(data["images"][i]["id"])
            self.imageID_to_url[self.image_ids[i]] = self.urls[i]
            
        # iterating through captions
            
        self.caption_ids = []
        self.captions = []
        self.image_ids_repetitions = []  
            
        
        self.captionID_to_caption = {}
        self.captionID_to_embedding = {}

        for a in range(len(data["annotations"])):
            self.caption_ids.append(data["annotations"][a]["id"])
            self.captions.append(data["annotations"][a]["caption"])
            self.image_ids_repetitions.append(data["annotations"][a]["image_id"])
                
            self.captionID_to_caption[self.caption_ids[a]] = self.captions[a]
#             self.captionID_to_embedding[self.caption_ids[a]] = embedding(self.captions[a])
            self.imageID_to_captions[self.image_ids_repetitions[a]].append(self.captions[a])        
            self.imageID_to_captionID[self.image_ids_repetitions[a]].append(self.caption_ids[a])

            

import numpy as np
import pickle

with open('resnet18_features.pkl', mode='rb') as file:
    imgID_to_descriptor = pickle.load(file)

def generate_triplet_set(image_id, cd):
    """
    Returns 10 triplets associated with a particular image id
    
    Parameters
    ----------
    image_id : int
        id associated with image
    all_image_ids : List
        all of image id's to pick bad image descriptors and caption from (excludes current image id)
    cd : CocoDataClass
        coco data class containing methods to query information
    
    Returns
    -------
    triplet_set : List[Tuple]
        10 triplets with good image descriptor, good caption, bad image descriptor
    """
    if image_id in imgID_to_descriptor:
        d_good = imgID_to_descriptor[image_id]  # get image descriptor vector given image id (name subject to change)
    else:
        return []
    good_caption_ids = np.random.choice(cd.imageID_to_captionID[image_id], size = 10)  # Generate caption ids given image id (randomly choose one each time)
    all_w_good = [cd.captionID_to_embedding[cap_id] for cap_id in good_caption_ids]  # convert caption id to embeddings
    d_bad = []  # Final 10 bad image descriptors
    
    for w_good in all_w_good:  # Iterate through each good caption to compare bad captions to
        other_img_id = np.random.choice(list(set(imgID_to_descriptor.keys()) - {image_id}), size=25)  # Generate 25 random other image ids
        other_w = [cd.captionID_to_embedding[np.random.choice(cd.imageID_to_captionID[img_id])] for img_id in other_img_id]  # Randomly pick caption given image id and get its embeddings
        w_bad_index = max([(w_good @ w_bad, i) for i, w_bad in enumerate(other_w)])[1]  # Dot product each other embedding with w_good, find biggest product, return that index
        d_bad += [imgID_to_descriptor[other_img_id[w_bad_index]]]
    
    return list(zip([d_good] * 10, all_w_good, d_bad)) # Zip up d_good, w_good, d_bad - 10 len-3 tuples


def generate_triplets(cd):
    """
    Returns Triplets of Good image descriptor, good caption, bad image descriptor
    
    Parameters
    ----------
    cd : CocoDataClass
        coco data class containing methods to query information
        
    Returns
    -------
    train : List[Tuple]
        4/5 of data - triplets containing good image descriptor, good caption, bad image descriptor
    validation : List[Tuple]
        1/5 of data - triplets containing good image descriptor, good caption, bad image descriptor
    """
    image_ids = cd.image_ids  # Get all image ids (subject to change based on attribute name)
    split_index = 4 * len(image_ids) // 5  # Split data at this index to train and validation
    idxs = np.arange(len(image_ids))  
    np.random.shuffle(idxs)  # Shuffle indices
    
    train_img_ids = [image_ids[x] for x in idxs[:split_index]]
    val_img_ids = [image_ids[x] for x in idxs[split_index:]]  # Validation set of image ids
    
    train = []  # Instantiate Final List of Triplets for train data
    validation = []  # Instantiate Final List of Triplets for validation data
    
    print (len(train_img_ids))
    for i, image_id in enumerate(train_img_ids[:100]):  # Go through each Train image id
        train += generate_triplet_set(image_id, cd)  # Generate and add triplet
    
    print (len(val_img_ids))
    for i, image_id in enumerate(val_img_ids[:100]):
        validation += generate_triplet_set(image_id, cd)
    
    return train, validation

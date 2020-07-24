import pickle
from coco_class import Coco
from Query_text_from_user import query_text
from img_search import search
from display_img import display

with open('embedding_database.pkl', mode='rb') as file:
    imgID_to_descriptor = pickle.load(file)
with open('FINAL_COCO_DB.pkl', mode='rb') as file:
    coco = pickle.load(file)

def run():
    query_emb = query_text(np.array(list(coco.word_to_idf.values())),
                           list(coco.word_to_idf.keys()))

    num_imgs = 4
    display(search(num_imgs, query_emb, imgID_to_descriptor), coco, cols=5, figsize=(10, 20))

import numpy as np

def search(k, query, id_to_semantic):
    """
    Computes the cosine distance between the query embedding and all image
    embeddings

    Parameters
    ----------
    k : int
        The number of closest images to be retrieved
    query : np.array - shape-(50,)
        The given query embedding
    semantic : dict
        A dictionary (length N) of image ids and respective embeddings (each
        being shape-(50,))
    Returns
    -------
    top_k : List[ints or floats or whatever the ids are]
    """

    ids = np.array(list(id_to_semantic.keys()))
    semantic_embs = np.swapaxes(np.array(list(id_to_semantic.values())), 0, 1)

    cos = np.matmul(query, semantic_embs) / (query * semantic_embs)

    top_k = []
    for j in range(k):
        i = np.argmax(cos)
        id = ids[np.argmax(cos)]
        top_k.append(id)
        cos = np.delete(cos, i)
        ids = np.delete(ids, i)

    return top_k

import io
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import math

__all__ = ["download_image"]

def download_image(img_url: str) -> Image:
    """Fetches an image from the web.

    Parameters
    ----------
    img_url : string
        The url of the image to fetch.

    Returns
    -------
    PIL.Image
        The image.
    """

    response = requests.get(img_url)
    return Image.open(io.BytesIO(response.content))


def display(img_ids: list, cd, cols=6, figsize=None):
    """ Fetches an image from the web.

    Parameters
    ----------
    img_ids : List[image IDs]
        List of respective image IDs to diplay.

    cd : CocoDataClass
        Coco data class containing image ID information

    cols : int
        Number of columns for the pictures to be displayed across (rows not
        included to avoid user indexing errors).

    figsize : tuple, len-(2)
        Figure dimension (width, height) in inches.
    """

    img_urls = []
    for id in img_ids:
        img_urls.append(cd.imageID_to_url[id])

    images = []
    for url in img_urls:
        images.append(download_image(url))

    rows = ceil(len(img_ids) / cols)
    axes = []
    fig = plt.figure(figsize=figsize)

    for i in range(len(img_urls)):
        axes.append(fig.add_subplot(rows, cols, i+1))
        axes[-1].set_xticks([])
        axes[-1].set_yticks([])
        plt.imshow(images[i])

    fig.tight_layout()
    plt.show()


import socket, warnings
try:
    socket.setdefaulttimeout(1)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('1.1.1.1', 53))
except socket.error as ex: raise Exception("STOP: No internet. Click '>|' in top right and set 'Internet' switch to on")



import os
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

if iskaggle:
    get_ipython().system('pip install -Uqq fastai duckduckgo_search')

from duckduckgo_search import ddg_images
from fastcore.all import *

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')
    ## L() is a fastai's implementation of python list with additional functionalities (.itemgot()) 
    ## The above line is similar to the line below and then indexing the return:
    #return ddg_images(term, max_results=max_images)
    #urls[0]['image']


urls = search_images('parrot photos', max_images=1)
urls[0]



from fastdownload import download_url
dest = 'parrot.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)

download_url(search_images('eagle photos', max_images=1)[0], 'eagle.jpg', show_progress=False)
Image.open('eagle.jpg').to_thumb(256,256)

searches = 'eagle','parrot'
path = Path('parrot_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} flying photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} in shade photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} in sun photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)


prediction, prediction_index, probs = learn.predict(PILImage.create('eagle.jpg'))
print(f"This is a/an '{prediction}'.")
print(f"Probability it's a/an {prediction}: {probs[prediction_index]:.4f}")


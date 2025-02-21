---
theme: ub
paginate: true
marp: true
math: katex
auto-scaling: true
header-includes:
  - \usepackage{algorithm2e}
---

<!-- _backgroundImage: "url('../slides/title.png')" -->
<!-- _paginate: skip -->

# EAS510 Basic of AI

<span class="subtitle">PyTorch Custom Datasets</span>

<div class="course-info">
  <p>MoWeFr 2:00PM-2:50PM</p>
  <p>Norton 209</p>
  <p>Instructor: <strong>Jue Guo</strong></p>
  <p>01/22/2025 - 05/06/2025</p>
</div>

---

## Custom Dataset

<div class = "columns">
<div>

A **custom dataset** is a collection of data relating to a specific problem you're working on.

- In essence, a **custom dataset** can be comprised of almost anything.

![](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pytorch-domain-libraries.png)

</div>
<div>

- if we were trying to build a model to classify whether or not a text-based review on a website was positive or negative, our custom dataset might be examples of existing customer reviews and their ratings.
- if we were trying to build a sound classification app, our custom dataset might be sound samples alongside their sample labels.
- if we were trying to build a recommendation system for customers purchasing things on our website, our custom dataset might be examples of products other people have bought.

PyTorch includes many existing functions to load in various custom datasets in the `TorchVision`, `TorchText`, `TorchAudio` and `TorchRec` domain libraries.

- But sometimes these existing functions may not be enough. In that case, we can always subclass `torch.utils.data.Dataset` and customize it to our liking.

Now fianlly we are gonna build something useful and what we want !!!

</div>
</div>

---

## What are we going to cover?

| **Topic** | **Contents** |
| ----- | ----- |
| **0. Import PyTorch and setup** | Load PyTorch and setup device-agnostic code. |
| **1. Get data** | Use a custom dataset of pizza, steak, and sushi images. |
| **2. Data preparation** | Understand and prepare the data. |
| **3. Transform data** | Transform images for model readiness. |
| **4. Load data with `ImageFolder`** | Use `ImageFolder` for standard image classification format. |
| **5. Custom `Dataset`** | Build a custom subclass of `torch.utils.data.Dataset`. |
| **6. Data augmentation** | Explore `torchvision`'s data augmentation functions. |
| **7. Model 0: TinyVGG without augmentation** | Build and train a model without data augmentation. |
| **8. Loss curves** | Analyze loss curves to check for underfitting or overfitting. |
| **9. Model 1: TinyVGG with augmentation** | Train a model with data augmentation. |
| **10. Compare results** | Compare loss curves and model performance. |
| **11. Predict on custom image** | Use the trained model to predict on new images. |

---

## Importing PyTorch and setting up device-agnostic code

```python
import torch
from torch import nn

# Note: this notebook requires torch >= 1.10.0
torch.__version__
```

```sh
'1.12.1+cu113'
```


And now let's follow best practice and setup device-agnostic code.

> **Note:** If you're using Google Colab, and you don't have a GPU turned on yet, it's now time to turn one on via `Runtime -> Change runtime type -> Hardware accelerator -> GPU`. If you do this, your runtime will likely reset and you'll have to run all of the cells above by going `Runtime -> Run before`.


```python
# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

```sh
'cuda'
```

---

## Get data and Data preparation

<div class = "columns">
<div>


We are teaching you how to build a custom dataset, so we need some data to work with.

The data we're going to be using is a subset of the [Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

- Food101 is popular computer vision benchmark as it contains 1000 images of 101 different kinds of foods, totaling 101,000 images (75,750 train and 25,250 test).

Instead of using all 101 classes, we're going to use a subset of 3 classes: pizza, steak, and sushi. This is for learning purposes and to speed up training time.

- You have to understand the data and prepare it for the model. 

Image classification format contains separate classes of images in separate directories titled with a particular class name. For example, all images of `pizza` are contained in the `pizza/` directory.

- This format is popular across many different image classification benchmarks, including [ImageNet](https://www.image-net.org/) (of the most popular computer vision benchmark datasets).
</div>

<div>

```sh
pizza_steak_sushi/ <- overall dataset folder
    train/ <- training images
        pizza/ <- class name as folder name
            image01.jpeg
            image02.jpeg
            ...
        steak/
            image24.jpeg
            image25.jpeg
            ...
        sushi/
            image37.jpeg
            ...
    test/ <- testing images
        pizza/
            image101.jpeg
            image102.jpeg
            ...
        steak/
            image154.jpeg
            image155.jpeg
            ...
        sushi/
            image167.jpeg
            ...
```

</div>
</div>

--- 

## Transform data

<div class = "columns">
<div>

The goal will be to **take this data storage structure and turn it into a dataset usable with PyTorch**.

> **Note:** The structure of the data you work with will vary depending on the problem you're working on. But the premise still remains: become one with the data, then find a way to best turn it into a dataset compatible with PyTorch.

We can use Python's in-built `os.walk()` to write a helper function that inspects our data directory by walking through subdirectories and counting the files present.

```python
import os
def walk_through_dir(dir_path):
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
```
</div>
<div>

```python
walk_through_dir(image_path)
```
```sh
There are 2 directories and 1 images in 'data/pizza_steak_sushi'.
There are 3 directories and 0 images in 'data/pizza_steak_sushi/test'.
There are 0 directories and 19 images in 'data/pizza_steak_sushi/test/steak'.
There are 0 directories and 31 images in 'data/pizza_steak_sushi/test/sushi'.
There are 0 directories and 25 images in 'data/pizza_steak_sushi/test/pizza'.
There are 3 directories and 0 images in 'data/pizza_steak_sushi/train'.
There are 0 directories and 75 images in 'data/pizza_steak_sushi/train/steak'.
There are 0 directories and 72 images in 'data/pizza_steak_sushi/train/sushi'.
There are 0 directories and 78 images in 'data/pizza_steak_sushi/train/pizza'.
```


It looks like we've got about 75 images per training class and 25 images per testing class.

- Remember, these images are subsets of the original Food101 dataset.

```python
# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir, test_dir
```
```sh
(PosixPath('data/pizza_steak_sushi/train'),
    PosixPath('data/pizza_steak_sushi/test'))
```

</div>
</div>

---

## Machine Learning Engineer Bash Command (Cheatsheet)

<div class = "columns">
<div>

**Navigating Directories**
- `ls`: List files and directories in the current directory.
- `cd <directory>`: Change to the specified directory.
- `pwd`: Print the current working directory.
  
**File Operations**
- `cp <source> <destination>`: Copy files or directories.
- `mv <source> <destination>`: Move files or directories.
- `rm <file>`: Remove files or directories.
- `rm -r <directory>`: Remove directories and their contents.
- `mkdir <directory>`: Create a new directory.
- `touch <file>`: Create a new file.
- `cat <file>`: Display the contents of a file.
- `head <file>`: Display the first few lines of a file.
- `tail <file>`: Display the last few lines of a file.
- `grep <pattern> <file>`: Search for a pattern in a file.
</div>

<div>

**Version Control**

- `git clone <repository>`: Clone a repository from a URL.
- `git add <file>`: Add a file to the staging area.
- `git commit -m "<message>"`: Commit changes to the repository.
- `git push`: Push changes to a remote repository.
- `git pull`: Pull changes from a remote repository.

**Networking**

- `ping <host>`: Send a ping request to a host.
- `curl <url>`: Download the contents of a URL.
- `wget <url>`: Download a file from a URL.

There is a lot more to learn, but these commands should get you started with the basics of working with the command line.

</div>
</div>

---

## Visualize the data

<div class = "columns">
<div>

Now in the spirit of the data explorer, it's time to *visualize, visualize, visualize!*

Let's write some code to:
1. Get all of the image paths using [`pathlib.Path.glob()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob) to find all of the files ending in `.jpg`. 
2. Pick a random image path using Python's [`random.choice()`](https://docs.python.org/3/library/random.html#random.choice).
3. Get the image class name using [`pathlib.Path.parent.stem`](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.parent).
4. And since we're working with images, we'll open the random image path using [`PIL.Image.open()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open) (PIL stands for Python Image Library).
5. We'll then show the image and print some metadata.

</div>
<div>

```python
import random
from PIL import Image

# Set seed
random.seed(42) # <- try changing this and see what happens

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")
img
```

</div>
</div>

---

## Visualize the data (cont.)

<div class = "columns">
<div>

```sh
Random image path: data/pizza_steak_sushi/test/pizza/2124579.jpg
Image class: pizza
Image height: 384
Image width: 512
```

<center>
<img src="../markdown_files/04_pytorch_custom_datasets_files/04_pytorch_custom_datasets_17_1.png" alt="Visualization" width="60%"> 
</center>

We can do the same with [`matplotlib.pyplot.imshow()`](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.imshow.html), except we have to convert the image to a NumPy array first.

</div>
<div>

```python
import numpy as np
import matplotlib.pyplot as plt

# Turn the image into an array
img_as_array = np.asarray(img)

# Plot the image with matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False);
```

</div>
</div>

---

## Transforming data

<div class = "columns">
<div>

Now we want to load our image data into PyTorch, before we can use our image data with PyTorch we need to:

1. Turn it into tensors (numerical representations of our images).
2. Turn it into a `torch.utils.data.Dataset` and subsequently a `torch.utils.data.DataLoader`, we'll call these `Dataset` and `DataLoader` for short.

<center>

| **Problem space** | **Pre-built Datasets and Functions** |
| ----- | ----- |
| **Vision** | [`torchvision.datasets`](https://pytorch.org/vision/stable/datasets.html) |
| **Audio** | [`torchaudio.datasets`](https://pytorch.org/audio/stable/datasets.html) |
| **Text** | [`torchtext.datasets`](https://pytorch.org/text/stable/datasets.html) |
| **Recommendation system** | [`torchrec.datasets`](https://pytorch.org/torchrec/torchrec.datasets.html) |

</center>

Since we're working with a vision problem, we'll be looking at `torchvision.datasets` for our data loading functions as well as [`torchvision.transforms`](https://pytorch.org/vision/stable/transforms.html) for preparing our data.

</div>
<div>

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

Let's see how we can transform data with `torchvision.transforms`.

</div>
</div>

---

## Transforming data with `torchvision.transforms`

<div class = "columns">
<div>

We've got folders of images but before we can use them with PyTorch, we need to convert them into tensors. One of the ways we can do this is by using the `torchvision.transforms` module.

- `torchvision.transforms` contains many pre-built methods for formatting images, turning them into tensors and even manipulating them for **data augmentation** (the practice of altering data to make it harder for a model to learn, we'll see this later on) purposes . 
  
To get experience with `torchvision.transforms`, let's write a series of transform steps that:
1. Resize the images using [`transforms.Resize()`](https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html#torchvision.transforms.Resize) (from about 512x512 to 64x64, the same shape as the images on the [CNN Explainer website](https://poloclub.github.io/cnn-explainer/)).
2. Flip our images randomly on the horizontal using [`transforms.RandomHorizontalFlip()`](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html#torchvision.transforms.RandomHorizontalFlip) (this could be considered a form of data augmentation because it will artificially change our image data).
3. Turn our images from a PIL image to a PyTorch tensor using [`transforms.ToTensor()`](https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor).

</div>
<div>

We can compile all of these steps using [`torchvision.transforms.Compose()`](https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html#torchvision.transforms.Compose).

```python
# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])
```

</div>
</div>

---

## Transforming data with `torchvision.transforms` (cont.)

<div class = "columns">
<div>

Now we've got a composition of transforms, let's write a function to try them out on various images.

```python
def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
```

</div>
<div>

```python
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_path_list, 
                        transform=data_transform, 
                        n=3)
```

</div>
</div>

---

## Transforming data with `torchvision.transforms` (cont.)

<div class = "columns">
<div>

Let's see the transformed images.

<center>

![png](../markdown_files/04_pytorch_custom_datasets_files/04_pytorch_custom_datasets_25_0.png)
![png](../markdown_files/04_pytorch_custom_datasets_files/04_pytorch_custom_datasets_25_1.png)

</center>

</div>
<div>

<center>

![png](../markdown_files/04_pytorch_custom_datasets_files/04_pytorch_custom_datasets_25_2.png)

</center>

We've now got a way to convert our images to tensors using `torchvision.transforms`.
- We also manipulate their size and orientation if needed (some models prefer images of different sizes and shapes).

Generally, the larger the shape of the image, the more information a model can recover.

- For example, an image of size `[256, 256, 3]` will have 16x more pixels than an image of size `[64, 64, 3]` (`(256*256*3)/(64*64*3)=16`).

However, the tradeoff is that more pixels requires more computations.

</div>
</div>

---

## Loading Image Data Using [`ImageFolder`](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder)

<div class = "columns">
<div>

Alright, time to turn our image data into a `Dataset` capable of being used with PyTorch.

- Since our data is in standard image classification format, we can use the class [`torchvision.datasets.ImageFolder`](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder). Where we can pass it the file path of a target image directory as well as a series of transforms we'd like to perform on our images.

Let's test it out on our data folders `train_dir` and `test_dir` passing in `transform=data_transform` to turn our images into tensors.

```python
# Use ImageFolder to create dataset(s)
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
```

</div>
<div>

```sh
Train data:
Dataset ImageFolder
    Number of datapoints: 225
    Root location: data/pizza_steak_sushi/train
    StandardTransform
Transform: Compose(
                Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=None)
                RandomHorizontalFlip(p=0.5)
                ToTensor()
            )
Test data:
Dataset ImageFolder
    Number of datapoints: 75
    Root location: data/pizza_steak_sushi/test
    StandardTransform
Transform: Compose(
                Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=None)
                RandomHorizontalFlip(p=0.5)
                ToTensor()
            )
```

Beautiful!

It looks like PyTorch has registered our `Dataset`'s.

</div>
</div>

---

## Loading Image Data Using [`ImageFolder`](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder) (cont.)

<div class = "columns">
<div>

Let's inspect them by checking out the `classes` and `class_to_idx` attributes as well as the lengths of our training and test sets.

```python
# Get class names as a list
class_names = train_data.classes
class_names
```

```sh
['pizza', 'steak', 'sushi']
```

```python
# Can also get class names as a dict
class_dict = train_data.class_to_idx
class_dict
```

```sh
{'pizza': 0, 'steak': 1, 'sushi': 2}
```

```python
# Check the lengths
len(train_data), len(test_data)
```

```sh
(225, 75)
```

</div>
<div>

Nice! Looks like we'll be able to use these to reference for later.

How about our images and labels?

How do they look?

We can index on our `train_data` and `test_data` `Dataset`'s to find samples and their target labels.


```python
img, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")
```

```sh
Image tensor:
... it is just bunch of jibberish numbers ...
Image shape: torch.Size([3, 64, 64])
Image datatype: torch.float32
Image label: 0
Label datatype: <class 'int'>
```
Our images are now in the form of a tensor (with shape `[3, 64, 64]`) and the labels are in the form of an integer relating to a specific class (as referenced by the `class_to_idx` attribute).

</div>
</div>

---

## Plot it out using `matplotlib`

<div class = "columns">
<div>

```sh
Image tensor:
... it is just bunch of jibberish numbers ...
Image shape: torch.Size([3, 64, 64])
Image datatype: torch.float32
Image label: 0
Label datatype: <class 'int'>
```

How about we plot a single image tensor using `matplotlib`?

- We'll first have to to permute (rearrange the order of its dimensions) so it's compatible.

Right now our image dimensions are in the format `CHW` (color channels, height, width) but `matplotlib` prefers `HWC` (height, width, color channels).

```python
# Rearrange the order of dimensions
img_permute = img.permute(1, 2, 0)

# Print out different shapes (before and after permute)
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")
plt.figure(figsize=(10, 7))
```
</div>
<div>

```python
plt.imshow(img.permute(1, 2, 0))
plt.axis("off")
plt.title(class_names[label], fontsize=14);
```
```sh
Original shape: torch.Size([3, 64, 64]) -> [color_channels, height, width]
Image permute shape: torch.Size([64, 64, 3]) -> [height, width, color_channels]
```
<center>
<img src="../markdown_files/04_pytorch_custom_datasets_files/04_pytorch_custom_datasets_36_1.png" alt="Visualization" width="50%">
</center>

</div>
</div>

---

## Turn loaded images into `DataLoader`'s

<div class = "columns">
<div>

We've got our images as PyTorch `Dataset`'s but now let's turn them into `DataLoader`'s.

- We'll do so using [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

- Turning our `Dataset`'s into `DataLoader`'s makes them iterable so a model can go through and learn the relationships between samples and targets (features and labels).

To keep things simple, we'll use a `batch_size=1` and `num_workers=1`.

*What is `num_wokers`?* It defines how many subprocesses will be created to load your data.

- Think of it like this, the higher value `num_workers` is set to, the more compute power PyTorch will use to load your data.

- Personally, I usually set it to the total number of CPUs on my machine via Python's [`os.cpu_count()`](https://docs.python.org/3/library/os.html#os.cpu_count). **This ensures the `DataLoader` recruits as many cores as possible to load data.**

</div>
<div>

> **Note:** There are more parameters you can get familiar with using `torch.utils.data.DataLoader` in the [PyTorch documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

```python
# Turn train and test Datasets into DataLoaders
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=1, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=1, 
                             shuffle=False) # don't usually need to shuffle testing data

train_dataloader, test_dataloader
```

```sh
(<torch.utils.data.dataloader.DataLoader at 0x7f7f3c1b3b50>,
 <torch.utils.data.dataloader.DataLoader at 0x7f7f3c1b3b80>)
```

Now our data is iterable. 10/10 :)

</div>
</div>

---

## Mis. stuff to note

<div class = "columns">
<div>

Let's try it out and check the shapes.

```python
img, label = next(iter(train_dataloader))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")
```
```sh
Image shape: torch.Size([1, 3, 64, 64]) -> [batch_size, color_channels, height, width]
Label shape: torch.Size([1])
```

We could now use these `DataLoader`'s with a training and testing loop to train a model.

- But before we do, let's look at another option to load images (or **almost any other kind of data**).

</div>
<div>

What if a pre-built `Dataset` creator like [`torchvision.datasets.ImageFolder()`](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder) didn't exist? Or one for your specific problem didn't exist?

But wait, what are the pros and cons of creating your own custom way to load `Dataset`'s?

| Pros of creating a custom `Dataset` | Cons of creating a custom `Dataset` |
| ----- | ----- |
| Can create a `Dataset` out of almost anything. | Even though you *could* create a `Dataset` out of almost anything, it doesn't mean it will work. | 
| Not limited to PyTorch pre-built `Dataset` functions. | Using a custom `Dataset` often results in writing more code, which could be prone to errors or performance issues. |

To see this in action, let's work towards replicating `torchvision.datasets.ImageFolder()` by subclassing `torch.utils.data.Dataset` (the base class for all `Dataset`'s in PyTorch). 
- But still **YOU NEED TO DO THIS ALMOST 90 PERCENT OF TIME!!!**

</div>
</div>

---

## Loading Image Data with a Custom `Dataset`

<div class = "columns">
<div>

We'll start by importing the modules we need:
* Python's `os` for dealing with directories (our data is stored in directories).
* Python's `pathlib` for dealing with filepaths (each of our images has a unique filepath).
* `torch` for all things PyTorch.
* PIL's `Image` class for loading images.
* `torch.utils.data.Dataset` to subclass and create our own custom `Dataset`.
* `torchvision.transforms` to turn our images into tensors.
* Various types from Python's `typing` module to add type hints to our code.

> **Note:** You can customize the following steps for your own dataset. The premise remains: write code to load your data in the format you'd like it.

</div>
<div>

Some dummy proof imports to get you started. 

```python
import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List
```

Recreate one by one ... 

Remember how our instances of `torchvision.datasets.ImageFolder()` allowed us to use the `classes` and `class_to_idx` attributes?


```python
# Instance of torchvision.datasets.ImageFolder()
train_data.classes, train_data.class_to_idx
```

```sh
(['pizza', 'steak', 'sushi'], {'pizza': 0, 'steak': 1, 'sushi': 2})
```

</div>
</div>

---

## Creating a helper function to get class names

<div class = "columns">
<div>

Let's write a helper function capable of creating a list of class names and a dictionary of class names and their indexes given a directory path.

**To do** so, we'll: *(You see this idea of todo list is very important in programming)*
1. Get the class names using `os.scandir()` to traverse a target directory (ideally the directory is in standard image classification format).
2. Raise an error if the class names aren't found (if this happens, there might be something wrong with the directory structure).
3. Turn the class names into a dictionary of numerical labels, one for each class.

Let's see a small example of step 1 before we write the full function.

</div>
<div>

```python
# Setup path for target directory
target_directory = train_dir
print(f"Target directory: {target_directory}")

# Get the class names from the target directory
class_names_found = sorted([entry.name for entry in list(os.scandir(image_path / "train"))])
print(f"Class names found: {class_names_found}")
```
```sh
Target directory: data/pizza_steak_sushi/train
Class names found: ['pizza', 'steak', 'sushi']
```
Good it works!! What is the next step??

</div>
</div>

---

## Creating a helper function to get class names (cont.)

<div class = "columns">
<div>

From now on always always keep in mind if there is any repetitive task, you should write a function for it.

```python
# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
```

</div>
<div>

```python
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
```

Looking good!

Now let's test out our `find_classes()` function.


```python
find_classes(train_dir)
```
```sh
(['pizza', 'steak', 'sushi'], {'pizza': 0, 'steak': 1, 'sushi': 2})
```

</div>
</div>

---

## Create a custom `Dataset` to replicate `ImageFolder`

<div class = "columns">
<div>

Now we're ready to build our own custom `Dataset`. We'll build one to replicate the functionality of `torchvision.datasets.ImageFolder()`. 

- This will be good practice, plus, it'll reveal a few of the required steps to make your own custom `Dataset`.

It'll be a fair bit of a code... but nothing we can't handle!

Let's break it down:
1. Subclass `torch.utils.data.Dataset`.
2. Initialize our subclass with a `targ_dir` parameter (the target data directory) and `transform` parameter (so we have the option to transform our data if needed).
3. Create several attributes for `paths` (the paths of our target images), `transform` (the transforms we might like to use, this can be `None`), `classes` and `class_to_idx` (from our `find_classes()` function).
4. Create a function to load images from file and return them, this could be using `PIL` or [`torchvision.io`](https://pytorch.org/vision/stable/io.html#image) (for input/output of vision data). 

</div>

<div>

5. Overwrite the `__len__` method of `torch.utils.data.Dataset` to return the number of samples in the `Dataset`, this is recommended but not required. This is so you can call `len(Dataset)`.
6. Overwrite the `__getitem__` method of `torch.utils.data.Dataset` to return a single sample from the `Dataset`, this is required.

</div>
</div>

---

## Create a custom `Dataset` to replicate `ImageFolder` (cont.)

<div class = "columns">
<div>

Let's start coding out:

```python
# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset

# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)
```
</div>
<div>

```python
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)
```

</div>
</div>

---

## Create a custom `Dataset` to replicate `ImageFolder` (cont.)

<div class = "columns">
<div>

You can organize all the code in `data_loader.py`; 

```python
# Augment train data
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
```

Let's turn our training images (contained in `train_dir`) and our testing images (contained in `test_dir`) into `Dataset`'s using our own `ImageFolderCustom` class.

```python
train_data_custom = ImageFolderCustom(targ_dir=train_dir, 
                                      transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, 
                                     transform=test_transforms)
train_data_custom, test_data_custom
```

</div>
<div>



```sh
(<__main__.ImageFolderCustom at 0x7f7f3c1b3b50>,
 <__main__.ImageFolderCustom at 0x7f7f3c1b3b80>)
```

No errors, did it work? Let's double check by try calling `len()` on our new `Dataset`'s and find the `classes` and `class_to_idx` attributes.

```python
len(train_data_custom), len(test_data_custom)
```

```sh
(225, 75)
```

```python
train_data_custom.classes
```

```sh
['pizza', 'steak', 'sushi']
```

```python
train_data_custom.class_to_idx
```

```sh
{'pizza': 0, 'steak': 1, 'sushi': 2}
```

looks like it works!!

</div>
</div>

---

## Create a function to display random images

How about we take it up a notch and plot some random images to test our `__getitem__` override? 

Let's create a helper function called `display_random_images()` that helps us visualize images in our `Dataset'`s.

Specifically, it'll:
1. Take in a `Dataset` and a number of other parameters such as `classes` (the names of our target classes), the number of images to display (`n`) and a random seed. 
2. To prevent the display getting out of hand, we'll cap `n` at 10 images.
3. Set the random seed for reproducible plots (if `seed` is set). 
4. Get a list of random sample indexes (we can use Python's `random.sample()` for this) to plot.
5. Setup a `matplotlib` plot.
6. Loop through the random sample indexes found in step 4 and plot them with `matplotlib`.
7. Make sure the sample images are of shape `HWC` (height, width, color channels) so we can plot them.

---

## Create a function to display random images (cont.)

<div class = "columns">
<div>

```python 
# 1. Take in a Dataset as well as a list of class names
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    
    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(16, 8))
```
</div>
<div>

```python
    # 6. Loop through samples and display random samples 
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
```

</div>
</div>

---

## Create a function to display random images (cont.)

<div class = "columns">
<div>

To test if it will work first on something that we know works.

```python
# Display random images from ImageFolder created Dataset
display_random_images(train_data, 
                      n=5, 
                      classes=class_names,
                      seed=None)
```

![png](../markdown_files/04_pytorch_custom_datasets_files/04_pytorch_custom_datasets_70_0.png)

Okay, our function itself works; And now with the `Dataset` we created with our own `ImageFolderCustom`.

```python
# Display random images from ImageFolderCustom Dataset
display_random_images(train_data_custom, 
                      n=12, 
                      classes=class_names,
                      seed=None) # Try setting the seed for reproducible images
```
```sh
For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.
```
</div>
<div>

![png](../markdown_files/04_pytorch_custom_datasets_files/04_pytorch_custom_datasets_72_1.png)
    



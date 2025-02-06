---
theme: ub
paginate: true
marp: true
math: katex
auto-scaling: true
---

<!-- _backgroundImage: "url('../slides/title.png')" -->
<!-- _paginate: skip -->

# EAS510 Basic of AI

<span class="subtitle">Course Logistics & PyTorch Fundamentals</span>

<div class="course-info">
  <p>MoWeFr 2:00PM-2:50PM</p>
  <p>Norton 209</p>
  <p>Instructor: <strong>Jue Guo</strong></p>
  <p>01/22/2025 - 05/06/2025</p>
</div>

---
<!-- _backgroundImage: "url('../slides/title.png')" -->
<!-- _paginate: skip -->

# Course Logistics


---

## Table of Contents

<div class="columns">
<div>

- [Course Logistics](#course-logistics)
  - [Important Information](#important-information)
  - [Course Schedule](#course-schedule)
- [Pytorch Overview](#pytorch-overview)
  - [Who Uses PyTorch?](#who-uses-pytorch)
  - [What we will cover?](#what-we-will-cover)
  - [What we will cover? (cont )](#what-we-will-cover-cont-)
  - [Introduction to tensors](#introduction-to-tensors)
  - [Creating Tensors](#creating-tensors)
  - [Scalar](#scalar)
  - [Vectors](#vectors)


</div>

<div>

- [Pytorch Overview](#pytorch-overview)
  - [Matrix](#matrix)
  - [Tensors](#tensors)
  - [Summary](#summary)
  - [Random Tensors](#random-tensors)
  - [Tensor datatypes](#tensor-datatypes)
  - [Getting information from tensors](#getting-information-from-tensors)
- [Manipulating Tensors](#manipulating-tensors)
  - [Basic Operations](#basic-operations)
  - [Matrix Multiplication (is all you need)](#matrix-multiplication-is-all-you-need)
  - [Finding the min, max, mean, sum, etc (aggregation)](#finding-the-min-max-mean-sum-etc-aggregation)
  - [Positional min/max](#positional-minmax)
  - [Reshaping, stacking, squeezing and unsqueezing](#reshaping-stacking-squeezing-and-unsqueezing)
  - [Indexing (selecting data from tensors)](#indexing-selecting-data-from-tensors)

</div>



---

## Important Information

<div class="columns">
<div>

- *First*, **sign up for piazza** https://piazza.com/buffalo/spring2025/eas510lecai1
  - ask me questions, answer other students' questions, and get help from your peers.
  - notification about project, assignment, and other important information will be posted on piazza.
- *Second*, **sign up for github** and follow me on https://github.com/COD1995?tab=repositories
  - keep track of the course materials, assignments, and projects.
  - you will also need github to keep track of your project.

</div>

<div>

| Percentage | Letter Grade | Percentage | Letter Grade |
|------------|--------------|------------|--------------|
| 95-100     | A            | 70-74      | C+           |
| 90-94      | A-           | 65-69      | C            |
| 85-89      | B+           | 60-64      | C-           |
| 80-84      | B            | 55-59      | D            |
| 75-79      | B-           | 0-54       | F            |

| Component               | Weight / Details              |
|-------------------------|-------------------------------|
| *Attendance*              | 10% (Random Pop Quiz)         |
| *Programming Assignment*  | 30% (2 PA)                   |
| *Midterm*               | 30%                          |
| *Group Project*          | 30%                          |
</div>


---

## Course Schedule

| Week(s) & Approx. Dates            | Topics Covered                                         |
|------------------------------------|-------------------------------------------------------|
| Week 1 and Week 2 (Jan 22 – Feb 4) | PyTorch Fundamentals, PyTorch Workflow Fundamentals   |
| Week 3 and Week 4 (Feb 5 – Feb 18) | PyTorch Neural Network Classification & Computer Vision |
| Week 5 and Week 6 (Feb 19 – Mar 3) | Custom Datasets, Going Modular, and Transfer Learning |
| Week 7 (Mar 4 – Mar 10)            | Midterm (Coverage: Weeks 1–5) and Catch Up            |
| Week 8 (Mar 11 – Mar 16)           | Experiment Tracking & Paper Replicating (start)       |
| Mar 17 – Mar 22                    | **Spring Recess (No Classes)**                        |
| Week 9 (Mar 24 – Mar 29)           | Experiment Tracking & Paper Replicating (continued)   |
| Week 10, 11, 12, and 13 (Mar 30 – Apr 26) | Model Deployment & Project Presentation           |
| Week 14 and Week 15 (Apr 27 – May 6) | Project Presentation                                |


---

<!-- _backgroundImage: "url('../slides/title.png')" -->
<!-- _paginate: skip -->
# Pytorch Overview 

---
## Who Uses PyTorch?

<div class="columns">
<div>

**What is PyTorch?**
- [PyTorch](https://pytorch.org/) is an open source machine learning and deep learning framework.
- PyTorch allows you to manipulate and process data and write machine learning algorithms using Python code.
  
**Industry Adoption**

- Adopted by **Meta**, **Tesla**, **Microsoft**, and **OpenAI**.  
- Powers AI for self-driving cars, products, and research.  
- Example: Tesla's self-driving models ([DevCon 2019](https://youtu.be/oBklltKXtDE), [AI Day 2021](https://youtu.be/j0z4FweCy4M?t=2904)).

</div>
<div>

**Diverse Applications**

- **Agriculture**: Enables computer vision on tractors.  
  [Learn More](https://medium.com/pytorch/ai-for-ag-production-machine-learning-for-agriculture-e8cfdb9849a1).  
- Widely used in **research and industry**.  

![PyTorch Usage](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-pytorch-being-used-across-research-and-industry.png)

</div>
</div>

---

## What we will cover?

| **Topic** | **Contents** |
| ----- | ----- |
| **Introduction to tensors** | Tensors are the basic building block of all of machine learning and deep learning. |
| **Creating tensors** | Tensors can represent almost any kind of data (images, words, tables of numbers). |
| **Getting information from tensors** | If you can put information into a tensor, you'll want to get it out too. |
| **Manipulating tensors** | Machine learning algorithms (like neural networks) involve manipulating tensors in many different ways such as adding, multiplying, combining. |

---

## What we will cover? (cont )

| **Topic** | **Contents** |
| ----- | ----- |
| **Dealing with tensor shapes** | One of the most common issues in machine learning is dealing with shape mismatches (trying to mix wrong shaped tensors with other tensors). |
| **Indexing on tensors** | If you've indexed on a Python list or NumPy array, it's very similar with tensors, except they can have far more dimensions. |
| **Mixing PyTorch tensors and NumPy** | PyTorch plays with tensors ([`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html)), NumPy likes arrays ([`np.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)) sometimes you'll want to mix and match these. |
| **Reproducibility** | Machine learning is very experimental and since it uses a lot of *randomness* to work, sometimes you'll want that *randomness* to not be so random. |
| **Running tensors on GPU** | GPUs (Graphics Processing Units) make your code faster, PyTorch makes it easy to run your code on GPUs. |

---

## Introduction to tensors

<div class="columns">
<div>

**What are Tensors?**

- Tensors are the fundamental building block of machine learning.  
- They represent data in a numerical way.  
- Example: An image as a tensor with shape `[3, 224, 224]`:  
  - `3`: Colour channels (red, green, blue).  
  - `224`: Height in pixels.  
  - `224`: Width in pixels.  

In tensor-speak, this tensor has three dimensions:  
- `Colour channels`  
- `Height`  
- `Width`

</div>
<div>

![Example of going from an input image to a tensor representation of the image](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-tensor-shape-example-of-image.png)

**Let's Code!**
- Learn more about tensors by coding them.  

</div>
</div>

---
## Creating Tensors

- PyTorch loves tensors, and they’re fundamental to machine learning.  
- Tensors represent data numerically, like an image as `[3, 224, 224]`:
  - `3`: Colour channels (red, green, blue).  
  - `224`: Height in pixels.  
  - `224`: Width in pixels.  

> **Note**: We'll focus on writing code and exploring PyTorch documentation to build familiarity.

**Let's start with scalars.**

---

## Scalar
<div class="columns">
<div>

- A scalar is a single number and in tensor-speak it's a zero dimension tensor.

    ```python 
    # Scalar
    scalar = torch.tensor(7)
    scalar
    ```

      tensor(7)

</div>
<div>

- `scalar.ndim` will return `0` because it's a scalar.
- What if we wanted to retrieve number from the tensor?
  
    ```python
    scalar.item()
    ```


      7 
</div>

---

## Vectors
<div class="columns">
<div>

- A vector is a single dimension tensor but can contain many numbers.
  - As in, you could have a vector [3, 2] to describe [bedrooms, bathrooms] in your house. Or you could have [3, 2, 2] to describe [bedrooms, bathrooms, car_parks] in your house.

    ```python
    # Vector
    vector = torch.tensor([7, 7])
    vector
    ```
        tensor([7, 7])
</div>

<div>

- `vector.ndim` will return `1` because it's a vector.
  - Trick, you can tell the number of dimensions a tensor in PyTorch has by the number of square brackets on the outside (`[`) and you only need to count one side.

- `vector.shape` will return `torch.Size([2])` because it's a vector with 2 elements.

</div>

---

## Matrix

<div class="columns">
<div>

- A matrix is a two-dimensional tensor.
  - You could have a matrix like `[[7, 8], [9, 10]]` to describe a 2x2 grid of numbers.

    ```python
    # Matrix
    matrix = torch.tensor([[7, 8], 
                           [9, 10]])
    matrix
    ```

        tensor([[ 7,  8],
                [ 9, 10]])

  - MATRIX has two dimensions (did you count the number of square brackets on the outside of one side?).
</div>

<div>

- What `shape` do you think the matrix tensor has?
  - `torch.Size([2, 2])` because it's a 2x2 matrix.

---

## Tensors

<div class="columns">

<div>

```python
# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
TENSOR
```

    tensor([[[1, 2, 3],
            [3, 6, 9],
            [2, 4, 5]]])

- tensors can represent almost anything
- what about its shape? `TENSOR.shape` will return `torch.Size([1, 3, 3])`

</div>

<div>

![alt text](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-pytorch-different-tensor-dimensions.png)

---

## Summary


| Name | What is it? | Number of dimensions | Lower or upper (usually/example) |
| ----- | ----- | ----- | ----- |
| **scalar** | a single number | 0 | Lower (`a`) |
| **vector** | a number with direction (e.g. wind speed with direction) but can also have many other numbers | 1 | Lower (`y`) |
| **matrix** | a 2-dimensional array of numbers | 2 | Upper (`Q`) |
| **tensor** | an n-dimensional array of numbers | can be any number, a 0-dimension tensor is a scalar, a 1-dimension tensor is a vector | Upper (`X`) |

---

## Random Tensors

<div class="columns">

<div>
As a data scientist, 

- you can define how the machine learning model starts (initialization), looks at data (representation) and updates (optimization) its random numbers. This is something we will discuss later in the course.

```python 
# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4))
random_tensor, random_tensor.dtype
```

    (tensor([[0.6541, 0.4807, 0.2162, 0.6168],
            [0.4428, 0.6608, 0.6194, 0.8620],
            [0.2795, 0.6055, 0.4958, 0.5483]]),
    torch.float32)

</div>

<div>
The flexibility of `torch.rand()` is that we can adjust the `size` to be whatever we want.

- For example, say you wanted a random tensor in the common image shape of `[224, 224, 3]` (`[height, width, color_channels`]).


```python
# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3))
random_image_size_tensor.shape, random_image_size_tensor.ndim
```
    (torch.Size([224, 224, 3]), 3)

*please refer to the class notes to see how to create 0 and 1s tensor*
</div>

---

## Tensor datatypes 


There are many different [tensor datatypes available in PyTorch](https://pytorch.org/docs/stable/tensors.html#data-types).
- Generally if you see `torch.cuda` anywhere, the tensor is being used for GPU (since Nvidia GPUs use a computing toolkit called CUDA).

<div class="columns">
<div>

  ```python
  # Default datatype for tensors is float32
  float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                                dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                                device=None, # defaults to None, which uses the default tensor type
                                requires_grad=False) # if True, operations performed on the tensor are recorded 

  float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device
  ```

    (torch.Size([3]), torch.float32, device='cpu')

</div>

<div>

```python
float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

float_16_tensor.dtype
```

The reason for all of these is to do with precision in computing.

---

## Getting information from tensors


<div class="columns">
<div>

- Once you've created a tensor, you'll want to get information from it.
  - `shape`: The length of each of the dimensions of the tensor.
  - `dtype`: The datatype of the tensor.
  - `device`: The device the tensor is stored on (e.g. `cpu` or `cuda`).

```python
# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU
```
</div>

<div>


    tensor([[0.4688, 0.0055, 0.8551, 0.0646],
            [0.6538, 0.5157, 0.4071, 0.2109],
            [0.9960, 0.3061, 0.9369, 0.7008]])
    Shape of tensor: torch.Size([3, 4])
    Datatype of tensor: torch.float32
    Device tensor is stored on: cpu


- _"what shape are my tensors? what datatype are they and where are they stored? what shape, what datatype, where where where"_

---

<!-- _backgroundImage: "url('../slides/title.png')" -->
<!-- _paginate: skip -->

# Manipulating Tensors

---

## Basic Operations

<div class="columns">
<div>

Let's start with a few of the fundamental operations, addition, subtraction, multiplication. 

```python
# Create a tensor of values and add a number to it
tensor = torch.tensor([1, 2, 3])
tensor + 10
```

    tensor([11, 12, 13])


```python
# Multiply it by 10
tensor * 10
```

    tensor([10, 20, 30])

_Notice how the tensor values above didn't end up being tensor(`[110, 120, 130]`), this is because the values inside the tensor don't change unless they're reassigned._

</div>

<div>

```python
# Subtract and reassign
tensor = tensor - 10
tensor
```


    tensor([-9, -8, -7])



PyTorch also has a bunch of built-in functions like `torch.mul()` (short for multiplication) and `torch.add()` to perform basic operations.

- However, it's more common to use the operator symbols like * instead of `torch.mul()`

---

## Matrix Multiplication (is all you need)
One of the most common operations in machine learning and deep learning algorithms (like neural networks) is [matrix multiplication](https://www.mathsisfun.com/algebra/matrix-multiplying.html).

<div class="columns">
<div>

The main two rules for matrix multiplication to remember are:

1. The **inner dimensions** must match:
  * `(3, 2) @ (3, 2)` won't work
  * `(2, 3) @ (3, 2)` will work
  * `(3, 2) @ (2, 3)` will work
2. The resulting matrix has the shape of the **outer dimensions**:
 * `(2, 3) @ (3, 2)` -> `(2, 2)`
 * `(3, 2) @ (2, 3)` -> `(3, 3)`

> **Note:** "`@`" in Python is the symbol for matrix multiplication.

> **Resource:** You can see all of the rules for matrix multiplication using `torch.matmul()` [in the PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.matmul.html).

</div>

<div>

For our `tensor` variable with values `[1, 2, 3]`:

| Operation | Calculation | Code |
| ----- | ----- | ----- |
| **Element-wise multiplication** | `[1*1, 2*2, 3*3]` = `[1, 4, 9]` | `tensor * tensor` |
| **Matrix multiplication** | `[1*1 + 2*2 + 3*3]` = `[14]` | `tensor.matmul(tensor)` |



```python
# Element-wise matrix multiplication
tensor * tensor
# Matrix multiplication
torch.matmul(tensor, tensor)
```

    tensor([1, 4, 9]), tensor(14)


---

## Finding the min, max, mean, sum, etc (aggregation)

Now we've seen a few ways to manipulate tensors, let's run through a few ways to aggregate them (go from more values to less values).

First we'll create a tensor and then find the max, min, mean and sum of it.

<div class="columns">
<div>

```python
# Create a tensor
x = torch.arange(0, 100, 10)
x
```
Now let's perform some aggregation.

```python
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")
```

    Minimum: 0
    Maximum: 90
    Mean: 45.0
    Sum: 450

</div>

<div>

> **Note:** You may find some methods such as `torch.mean()` require tensors to be in `torch.float32` (the most common) or another specific datatype, otherwise the operation will fail.

You can also do the same as above with `torch` methods.


```python
torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x)
```

    (tensor(90), tensor(0), tensor(45.), tensor(450))

---

## Positional min/max

<div class="columns">
<div>

You can also find the index of a tensor where the max or minimum occurs with [`torch.argmax()`](https://pytorch.org/docs/stable/generated/torch.argmax.html) and [`torch.argmin()`](https://pytorch.org/docs/stable/generated/torch.argmin.html) respectively.

This is helpful incase you just want the position where the highest (or lowest) value is and not the actual value itself (we'll see this in a later section when using the [softmax activation function](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)).

```python
# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")
```
</div>
<div>

    Tensor: tensor([10, 20, 30, 40, 50, 60, 70, 80, 90])
    Index where max value occurs: 8
    Index where min value occurs: 0


On the side note, you can **change tensor datatype**: 
- If one tensor is in `torch.float64` and another is in `torch.float32`, you might run into some errors.

- You can change the datatypes of tensors using [`torch.Tensor.type(dtype=None)`](https://pytorch.org/docs/stable/generated/torch.Tensor.type.html) where the `dtype` parameter is the datatype you'd like to use.

```python
# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
tensor.dtype # torch.float32

# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
tensor_float16
```

</div>

---

## Reshaping, stacking, squeezing and unsqueezing

It's difficult to convey the usefulness of these operations in a slide without hands-on machine learning experience. Consider the following operations as a reference.

| Method | One-line description |
| ----- | ----- |
| [`torch.reshape(input, shape)`](https://pytorch.org/docs/stable/generated/torch.reshape.html#torch.reshape) | Reshapes `input` to `shape` (if compatible), can also use `torch.Tensor.reshape()`. |
| [`Tensor.view(shape)`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) | Returns a view of the original tensor in a different `shape` but shares the same data as the original tensor. |
| [`torch.stack(tensors, dim=0)`](https://pytorch.org/docs/1.9.1/generated/torch.stack.html) | Concatenates a sequence of `tensors` along a new dimension (`dim`), all `tensors` must be same size. |
| [`torch.squeeze(input)`](https://pytorch.org/docs/stable/generated/torch.squeeze.html) | Squeezes `input` to remove all the dimenions with value `1`. |
| [`torch.unsqueeze(input, dim)`](https://pytorch.org/docs/1.9.1/generated/torch.unsqueeze.html) | Returns `input` with a dimension value of `1` added at `dim`. |
| [`torch.permute(input, dims)`](https://pytorch.org/docs/stable/generated/torch.permute.html) | Returns a *view* of the original `input` with its dimensions permuted (rearranged) to `dims`. |

---

## Indexing (selecting data from tensors)

Sometimes you'll want to select specific data from tensors (for example, only the first column or second row).

<div class="columns">
<div>

```python
# Create a tensor
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape
```




    (tensor([[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]]),
     torch.Size([1, 3, 3]))



Indexing values goes outer dimension -> inner dimension (check out the square brackets).


```python
# Let's index bracket by bracket
print(f"First square bracket:\n{x[0]}")
print(f"Second square bracket: {x[0][0]}")
print(f"Third square bracket: {x[0][0][0]}")
```
</div>
<div>

    First square bracket:
    tensor([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
    Second square bracket: tensor([1, 2, 3])
    Third square bracket: 1

You can also use `:` to specify "all values in this dimension" and then use a comma (`,`) to add another dimension.


```python
# Get all values of 0th dimension and the 0 index of 1st dimension
x[:, 0]
```




    tensor([[1, 2, 3]])









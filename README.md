# Object-Oriented Programming for AI 2023-24 - final project requirements

In this final project, you will be required to implement a compex application in Python using the OOP concept explained in the course.
The project will need to contain the following elements:

* An implementation of a hierarchy of classes to manage datasets, mainly with Machine Learning as an end goal
  * A dataset wrapper, called `Dataloader`, which handles the creation of batches of data for training a model.
* An implementation of a stochastic minimzation algorithm to optimize the parameters of a model. This algorithm will need to interact with the dataset classes
* An implementation of a model class, which will need to interact with the dataset classes

The deadline for the assignment is set on **January 20th, 2024 23:59 CET**.

The assignment will need to be produced as a **private** GitHub repository with the following structure:

* A `README.md` file. The readme is to function as a submission report and a documentation for the usage of your code.
* A `requirements.txt` file, listing all the dependencies of your project. This is to be formatted according to the [pip requirements file format](https://note.nkmk.me/en/python-pip-install-requirements/).
* A folder containing all of your code. The name of the folder is up to you, but it should be something descriptive of the code functionality.


## Datasets

You will need to implement a hierarchy of classes to manage datasets.
The implementation will need to contain a base class, defining all the methods and attributes in common to all of the datasets.
Datasets should have two variants: one including both data proper and labels, the other including only the data proper (which can be used, e.g., for testing when there are no labels available).

All the datasets should include a `root` attribute, which identifies the root location where the data is stored into the disk.
Each data point (e.g., an image, an audio file) should be stored as a single file in the disk, with the name of the file uniquely identifying the data point.

Datasets can be both for regression and classification.

* In the case of **regression**, the data should be stored on disk in the `root` folder (and not any subfolder), with the labels stored in a separate file outside of the `root` folder.
The labels should be stored in a single file **outside of `root`**, with the file in a `.csv` format.
Each line contains information about a data point: the first column contains the filename and the second column containing the corresponding label.
* In the case of **classification**, there are two possible formats:
  1. As in the case of regression, the data is stored in a single folder, with the labels stored in a separate file outside of the `root` folder, in a `.csv` file formatted as above.
  2. The data is stored in a folder hierarchy, with each subfolder containing the data for a single class. For instance, if we have three categories, the data is stored in a folder hierarchy as follows:
   
        ```
        root
        ├── class_1
        │   ├── data_1
        │   ├── data_2
        │   └── data_3
        ├── class_2
        │   ├── data_4
        │   ├── data_5
        │   └── data_6
        └── class_3
            ├── data_7
            ├── data_8
            └── data_9
        ```
        Notice that, in this specific configuration, you don't need a file with labels, since the labels are already encoded in the folder hierarchy.

        You should account for both of these configurations in your implementation.

The datasets should be able to load the data from the disk, both in a **lazy** and in an **eager** fashion.
The eager implementation should **load all the data into memory at once**, while the lazy implementation should store only the data path and **load the data from the disk only when needed**.

In all the cases, the data should be accessed using the subsetting operator (e.g., `dataset[i]`), which should return (using the data structure you prefer) the data at the specified index.
In the case of a dataset with labels, you should return both the data and the corresponding label.
If the dataset has no labels, you should return only the data proper.

The datasets should have a method for splitting the data into training and test sets.
The user should be able to split the data by specifying the percentage of data to be used for training.
This function should return two datasets as output.

You should create at least two datasets with these characteristics, one handling **images**, the other handling **audio files**.
For handling images, you can load them using one of many Python libraries for image processing, such as [Pillow](https://pillow.readthedocs.io/en/stable/) or [OpenCV](https://opencv.org/).
Notice that OpenCV, while faster than Pillow, is using the BGR format for images, while Pillow is using the RGB format.
You should be careful that images loaded into the datasets are in the RGB format.

For handling audio files, you can use [Librosa](https://librosa.org/doc/latest/index.html), which is a Python library for audio processing.

### Dataloader class

The Dataloader will be constructed on top of a dataset, and will be responsible for creating batches of data for training a model using stochastic iterative methods.

The Dataloader should be able to create batches of data of a specified size.
In addition, the user should specify whether they want the batches to be created in a **random** or in a **sequential** fashion:

* In the case of **random** batches, the dataloader should create batches of data by randomly shuffling the order of the data points and then creating batches of the specified size.
* In the case of **sequential** batches, no shuffling should be performed, and the batches should be created in the original order of the data points.

Notice that, if `dataset_size // batch_size != 0` (i.e., the batch size is not a divisor of the dataset size), the last batch will be smaller than the specified batch size. You should let the user decide whether to use or to discard this last batch in case.

Within the dataloader, the batches are to be created only using the **indices** of the data points, and the data composing the batch should be loaded from the disk only when needed using an iterator.

If passed as argument to the `len` method, the dataloader should return the number of batches that can be created from the dataset with the specific batch size.

### Data preprocessing

Create at least three classes for data preprocessing:
1. One class for normalizing a dataset of images in the 0-1 range. Note: images are encoded as arrays with values in the 0-255 range.
2. 

## Extra (up to 1 point): inference through a pretrained neural network 

If you feel like trying your hand at something more complex, you can try to implement a generic `Model` class wrapping a pre-existing model for image or audio classification/regression.
The model should be able to perform inference on one or more data points of a dataset, and return the prediction as output.

To do so, you can use pretrained neural networks on popular image benchmarks.
The usage of the dataset ImageNet is not recommended, since it is too large to be used in this project, and it is currently unavailable for download in legal manners.
You can use, for instance, the [Imagenette](https://github.com/fastai/imagenette) dataset. There are several available pretrained models on Imagenette, which you can find on GitHub with the `imagenette` topic: https://github.com/topics/imagenette.

With reference to this ResNet18 implementation: https://github.com/GeorgeMLP/imagenette-classification, to make it work with your model, you will need to:

   1. [Install PyTorch and Torchvision](https://pytorch.org/get-started/locally/)
   2. Create the neural network (which will be wrapped by the `Model` class):

        ```python
        import torch
        from torchvision import models
        
        model = models.resnet18()
        # this architecture has 1000 output classes (ImageNet)
        # we need to replace the classification head to match
        # the number of classes in Imagenette (10)
        # replace classification head to match 10 output classes
        model.fc = torch.nn.Linear(512, 10)

        state_dict = torch.load("path/to/weights.pth", map_location="cpu")
        # in this case, state dict contains three keys:
        # 'net', 'optimizer', 'epoch'
        # we have to select the weights which are contained
        # in the 'net' key
        model.load_state_dict(state_dict['net'])
        ```

      You have succesfully loaded the pretrained weights into the model! Now you're ready to do inference!

   3. Load the Imagenette dataset using your dataset implementation & create a dataloader
   4. Before running the inference, remember to set the model in evaluation mode:

      ```python
      model.eval()
      ```

   5. Remember, before passing the data to the model, to apply the preprocessing pipeline indicated in the repo: https://github.com/GeorgeMLP/imagenette-classification/blob/master/Training%20Model.py, see lines 25-30.

# Demo

You should create a demo script, which will be used to show the functionality of your code.
The demo script should be able to:

* Showcase the funcionality of the datasets:
  * One lazy and one eager dataset
  * One dataset with labels and one without labels
  * One dataset with classification labels (using the ) and one with regression labels
  * 

# Generic indications

Remind to always make use of the OOP concepts explained in the course:

* Always implement encapsulation accordingly. All the attributes should be private, and the user should be able to access them only through getters and setters. The usage of the `@property` decorator is preferrable. Motivate any choice of public methods and attributes in the report.
* 
# Classification Training Sample Code

##About the Project
This project provides a sample implementation of a computer vision classification task using deep learning models. The code is designed to be easily configurable, allowing users to experiment with different training and data pipelines settings.
The project includes code for data preprocessing, model training and model evaluation. Users can configure different parameters for data pipelines with different model architectures and hyperparameters.
In the future, the project will be updated with additional configuration options. The goal is to provide a flexible and user-friendly codebase for exploring computer vision classification tasks.

## Installation and Setup
* Install python version 3.* on your local computer
* Run ```pip install -r requirements.txt ``` 

## Usage
### Data 
This Project assumes that the input data is stored in a specific directory structure. Here "0", "1", ..., and "n" represents different categories of images that you want to classify.
 
```
data_directory/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── n/
└── test/
    ├── 0/
    ├── 1/
    ├── ...
    └── n/

```
Before training the model, you will need to create CSV files for training and validation datasets. To do this, run the following commands:
```
python data_preparing.py --path "path/to/data" --name "train"
python data_preparing.py --path "path/to/data" --name "test"
```
These scripts will preprocess the images in the "{data_directory}/train" and "{data_directory}/test" directories and create two CSV files: 'train.csv' and 'test.csv'. These CSV files will contain the paths to the images files and their corresponding class labels. 

### Training
To train the model, run the following command:
```python main.py --config "path/to/config/file" ```
This will start the training process using the configuration settings in the "config.yaml" file. the "config.yaml" file contains all the hyperparameters and configuration settings for the training process. 
You can modify the "config.yaml" file to experiment with different settings and see how they affect the performance  of the model. Note that you should only modify the configuration settings in the "config.yaml" file and not in the code itself.


#### Configurations need to be explained
* ["classes"]: 
The *"classes"* option in the *"config.yaml"* file is used to specify the names of the classes that you want to classify. For example, if you are working on a classification task with three classes named *"cat"*, *"dog"*, and *"bird"*, your *"classes"* directory should look like this:
```
classes: {"0": "cat", "1":"dog", "2":"bird"}
```
* *["train"]["model"]*:
The *"model"* option in *"config.yaml"* file is used to specify the architecture of the model that you want to train. It should be set to one of the model names available in the **"timm"** library, which provides a wide range of state-of-the-art computer vision models.
To set *"model"* option, you can refer to the list of available models in the *"timm"* library documentation. You can find the documentation for *"timm"* at the following link: [timm library](https://github.com/huggingface/pytorch-image-models)

* *["train"]["mixed_precision"]*:
The *"mixed_precision"* option in the *"config.yaml"* file is used to enable mixed precision training. Mixed precision training is a technique for accelerating the training of deep neural networks by using lower-precision floating-point numbers to represent the network parameters and activations. To use mixed precision training, you can refer to the implementation in the [Pytorch mixed precision implementation](https://pytorch.org/docs/stable/notes/amp_examples.html)

* *["optimizer"]["name"]*:
The *"optimizer"* option in *"config.yaml"* file is used to specify the optimizer for training. Pytorch provides several built-in optimizer algorithms that can be used for training deep neural networks. Two common optimizers are Adam and SGD, both of which are supported by PyTorch. 
To use Adam or SGD, you can set *["optimizer"]["name"]* option in *"config.yaml"* file to *"adam"* or *"sgd"*. 
AdaBelief is a state-of-the-art optimizer algorithm that builds on top of the popular Adam optimizer. To use Adabelief, you can refer to the implementation in the [adabelief optimizer](https://github.com/juntang-zhuang/Adabelief-Optimizer) which implementation of this optimizer.
To use Adabelief optimizer you can set ["optimizer"]["name"] option in *"config.yaml"* file to *"adabelief"*.

* *["optimizer"]["sam"]*:
The "sam" option in the *"config.yaml"* file is used to enable the Sharpness-Aware Minimization (SAM) optimizer. SAM is a technique for optimizing deep neural networks that has been shown to improve convergence speed and generalization performance. To use the SAM optimizer, you can refer to the implementation in the [sam optimizer](https://github.com/davda54/sam) repository.

* *["optimizer"]["scheduler"]*:
The *"scheduler"* option in the *"config.yaml"* file is used to specify the learning rate scheduler used during training. To use the Cosine Annealing with Warm Restarts learning rate scheduler, you can refer to the implementation in the [Cosine Annealing with Warmup for Pytorch](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup) repository, which provides an implementation of this scheduler. 

#### Training Process Results
During the training process, the model's performance is monitored using various metrics, including accuracy, precision, recall, and F1 score. The results of the training process are logged to both a TensorBoard log file and a text log file. 

##### TensorBoard Log File
The TensorBoard log file is used to visualize the training progress in real-time. To view the TensorBoard log file, you can run the following command:
```
tensorboard --logdir=runs
```
This will start a TensorBoard server that you can access in your web browser.
 
##### Text Log File
The text log file contains detailed information about the training process, including the training and validation loss, accuracy, and F1 score. The log file is saved to the *'runs'* directory in the project's root directory.

##### Checkpoints
During training, the best performing model on test dataSet based on F1 score is saved in the runs directory. You can find the checkpoint file in the *runs/{time_of_run}/best_ckpt.pt* directory. This checkpoint file can be used to load the model and continue training or to make predictions on new data.

That's it! With this information, users should be able to view the training progress using TensorBoard, access the detailed training logs, and find the best performing model checkpoint.

### Updating the Repository
As new configurations become available or changes are made to the existing configurations, this repository will be updated to reflect those changes.

If you have a specific configuration that you would like to see added to the repository, please create an issue and describe the configuration in as much detail as possible. We will do our best to incorporate your suggestions into the repository.





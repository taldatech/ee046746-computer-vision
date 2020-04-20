# ee046746-computer-vision
Jupyter Notebook tutorials for the Technion's EE 046746 Computer Vision course

<img src="https://github.com/taldatech/ee046746-computer-vision/blob/master/assets/tut_track_anim.gif" width="400">

- [ee046746-computer-vision](#ee046746-computer-vision)
  * [Running The Notebooks](#running-the-notebooks)
    + [Running Online](#running-online)
    + [Running Locally](#running-locally)
  * [Agenda](#agenda)
  * [Installation Instructions](#installation-instructions)
    + [Libraries to Install](#libraries-to-install)

## Running The Notebooks
You can view the tutorials online or download and run locally.

### Running Online

|Service      | Usage |
|-------------|---------|
|Jupyter Nbviewer| Render and view the notebooks (can not edit) |
|Binder| Render, view and edit the notebooks (limited time) |
|Google Colab| Render, view, edit and save the notebooks to Google Drive (limited time) |


Jupyter Nbviewer:

[![nbviewer](https://jupyter.org/assets/main-logo.svg)](https://nbviewer.jupyter.org/github/taldatech/ee046746-computer-vision/tree/master/)


Press on the "Open in Colab" button below to use Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taldatech/ee046746-computer-vision)

Or press on the "launch binder" button below to launch in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/taldatech/ee046746-computer-vision/master)

Note: creating the Binder instance takes about ~5-10 minutes, so be patient

### Running Locally

Press "Download ZIP" under the green button `Clone or download` or use `git` to clone the repository using the 
following command: `git clone https://github.com/taldatech/ee046746-computer-vision.git` (in cmd/PowerShell in Windows or in the Terminal in Linux/Mac)

Open the folder in Jupyter Notebook (it is recommended to use Anaconda). Installation instructions can be found in `Setting Up The Working Environment.pdf`.



## Agenda

|File       | Topics Covered |
|----------------|---------|
|`Setting Up The Working Environment.pdf`| Guide for installing Anaconda locally with Python 3 and PyTorch, integration with PyCharm and using GPU on Google Colab |
|`ee046746_tut_01_intro_image_processing_python.ipynb\pdf`| Python basics: NumPy, Matplotlib, OpenCV basics: Reading and Writing Images, Basic Image Manipulations, Image Processing 101: Thresholding, Blurring |
|`ee046746_tut_01_2_deep_learning_pytorch_basics.ipynb\pdf`| Deep Learning and PyTorch basics, MNIST, Fashion-MNIST, MULTI-layer Perceptron (MLP), Fully-Connected (FC) |
|`ee046746_tut_02_03_convolutional_neural_networks.ipynb\pdf`| 2D Convolution (Cross-corelation), Convolution-based Classification, Convolutional Neural Networks (CNNs), Regularization and Overfitting, Dropout, Data Augmentation, CIFAR-10 dataset, Visualizing Filters, The history of CNNs, Applications of CNNs, The problems with CNNs (adversarial attacks, poor generalization, fairness-undesirable biases)  |
|`ee046746_tut_02_03_appndx_visualizing_cnn_filters.ipynb\pdf`| Appendix - How to visulaize CNN filters and filter activations given image with PyTorch |
|`ee046746_tut_04_edge_and_line_detection.ipynb\pdf`| Edge and Line detection: Canny, Hough transform, RANSAC, and SCNN |

## Installation Instructions

For the complete guide, with step-by-step images, please consult `Setting Up The Working Environment.pdf`

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/distribution/
2. Create a new environment for the course:
In Windows open `Anaconda Prompt` from the start menu, in Mac/Linux open the terminal and run `conda create --name torch`. Full guide at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
3. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate torch`
4. Install the required libraries according to the table below (to search for a specific library and the corresponding command you can also look at https://anaconda.org/)

### Libraries to Install

|Library         | Command to Run |
|----------------|---------|
|`Jupyter Notebook`|  `conda install -c conda-forge notebook`|
|`numpy`|  `conda install -c conda-forge numpy`|
|`matplotlib`|  `conda install -c conda-forge matplotlib`|
|`scipy`| `conda install -c anaconda scipy `|
|`scikit-learn`|  `conda install -c conda-forge scikit-learn`|
|`opencv`|  `conda install -c conda-forge opencv`|
|`pytorch` (cpu)| `conda install pytorch torchvision cpuonly -c pytorch` |
|`pytorch` (gpu)| `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch` |


5. To open the notbooks, open Anancinda Navigator or run `jupyter notebook` in the terminal (or `Anaconda Prompt` in Windows) while the `torch` environment is activated.
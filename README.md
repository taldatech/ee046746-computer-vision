# ee046746-computer-vision

<h1 align="center">
  <br>
Technion EE 046746 - Computer Vision
  <br>
  <img src="https://raw.githubusercontent.com/taldatech/ee046746-computer-vision/master/assets/tut_track_anim.gif" height="200">
</h1>
  <p align="center">
    <a href="https://taldatech.github.io">Tal Daniel</a> •
    <a href="https://eliasnehme.github.io/">Elias Nehme</a> •
    <a href="https://github.com/dahliau">Dalia Urbach</a> •
  <a href="https://webee.technion.ac.il/people/anat.levin/">Anat Levin</a>
  </p>

Jupyter Notebook tutorials for the Technion's EE 046746 Computer Vision course

Previous semesters: <a href="https://github.com/taldatech/ee046746-computer-vision/tree/spring20">Spring 2020</a>

<h4 align="center">
    <a href="https://colab.research.google.com/github/taldatech/ee046746-computer-vision"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <a href="https://nbviewer.jupyter.org/github/taldatech/ee046746-computer-vision/tree/master/"><img src="https://jupyter.org/assets/main-logo.svg" alt="Open In NBViewer"/></a>
    <a href="https://mybinder.org/v2/gh/taldatech/ee046746-computer-vision/master"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a>

</h4>

- [ee046746-computer-vision](#ee046746-computer-vision)
  * [Agenda](#agenda)
  * [Running The Notebooks](#running-the-notebooks)
    + [Running Online](#running-online)
    + [Running Locally](#running-locally)
  * [Installation Instructions](#installation-instructions)
    + [Libraries to Install](#libraries-to-install)
  

## Agenda

|File       | Topics Covered |
|----------------|---------|
|`Setting Up The Working Environment.pdf`| Guide for installing Anaconda locally with Python 3 and PyTorch, integration with PyCharm and using GPU on Google Colab |
|`ee046746_tut_01_intro_image_processing_python.ipynb\pdf`| Python basics: NumPy, Matplotlib, OpenCV basics: Reading and Writing Images, Basic Image Manipulations, Image Processing 101: Thresholding, Blurring |
|`ee046746_tut_01_2_deep_learning_pytorch_basics.ipynb\pdf`| Deep Learning and PyTorch basics, MNIST, Fashion-MNIST, MULTI-layer Perceptron (MLP), Fully-Connected (FC) |
|`ee046746_tut_02_edge_and_line_detection.ipynb\pdf`| Edge and Line detection, Canny, Hough transform, RANSAC, and SCNN |
|`ee046746_tut_03_04_convolutional_neural_networks.ipynb\pdf`| 2D Convolution (Cross-corelation), Convolution-based Classification, Convolutional Neural Networks (CNNs), Regularization and Overfitting, Dropout, Data Augmentation, CIFAR-10 dataset, Visualizing Filters, The history of CNNs, Applications of CNNs, The problems with CNNs (adversarial attacks, poor generalization, fairness-undesirable biases) |
|`ee046746_tut_03_04_appndx_visualizing_cnn_filters.ipynb\pdf`| Appendix - How to visualize CNN filters and filter activations given image with PyTorch |
|`ee046746_tut_05_deep_semantic_segmentation.ipynb\pdf`| Semantic Segmentation, Intersection over Union (IoU), Average Precision (AP), PASCAL Visual Object Classes, Common Objects in COntext (COCO), Fully Convolutional Network (FCN),Up-Convolution / Transposed-Convolution, Skip connections, Pyramid Scene Parsing Network (PSPNet), 1x1 convolution, Mask R-CNN, DeepLab, Atrous convolution, Conditional Random Field (CRF) |
|`ee046746_tut_06_generative_adversarial_networks_gan.ipynb\pdf`| Generative Adversarial Network (GAN), Explicit/Implicit density estimation, Nash Equilibrium, Mode Collapse, Vanishing/Diminishing Gradient, Conditional GANs, WGAN, EBGAN, BEGAN, Tips for Training GANs, Pix2Pix, CycleGAN |
|`ee046746_tut_07_alignment.ipynb\pdf`| Feature Matching, Parametric Transformations, Image Warping, Image Blending, Panorama Stitching, Kornia |
|`ee046746_tut_08_deep_uncertainty.ipynb\pdf`| Need for Uncertainty, Epistemic and Aleatoric Uncertainty, Logelikihood Modelling, Bayesian Neural Networks, Dropout, Evidental Deep Learning |
|`ee046746_tut_09_deep_object_detection.ipynb\pdf`|Deep Object Detection, Localization, Sliding Windows, IoU, AP, Region-based Convolutional Neural Networks (R-CNN) Family, Fast/er R-CNN, Selective Search, Non-Maximum Supression (NMS), Region of Interest Pooling Layer (RoI), Region Proposal Network (RPN), Anchor boxes, Detectron2, You Only Look Once (YOLO) Family, YOLO V1-V4, Single Shot Multibox Detection (SSD) |
|`ee046746_tut_10_geometry_review.ipynb\pdf`| Camera Models, Camera Matrix, Intrinsic and Extrinsic Parameters, Distortion Models, Camera Calibration, Homography Edge Cases, Epipolar Geometry, Essential/Fundamental Matrix, 8-point Algorithm |

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



## Installation Instructions

For the complete guide, with step-by-step images, please consult `Setting Up The Working Environment.pdf`

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/products/individual
2. Install the basic packages using the provided `environment.yml` file by running: `conda env create -f environment.yml` which will create a new conda environment named `deep_learn`. If you did this, you will only need to install PyTorch, see the table below.
3. Alternatively, you can create a new environment for the course and install packages from scratch:
In Windows open `Anaconda Prompt` from the start menu, in Mac/Linux open the terminal and run `conda create --name deep_learn`. Full guide at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
4. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate deep_learn`
5. Install the required libraries according to the table below (to search for a specific library and the corresponding command you can also look at https://anaconda.org/)

### Libraries to Install

|Library         | Command to Run |
|----------------|---------|
|`Jupyter Notebook`|  `conda install -c conda-forge notebook`|
|`numpy`|  `conda install -c conda-forge numpy`|
|`matplotlib`|  `conda install -c conda-forge matplotlib`|
|`scipy`| `conda install -c anaconda scipy `|
|`scikit-learn`|  `conda install -c conda-forge scikit-learn`|
|`tqdm`| `conda install -c conda-forge tqdm`|
|`opencv`| `conda install -c conda-forge opencv`|
|`pytorch` (cpu)| `conda install pytorch torchvision torchaudio cpuonly -c pytorch` |
|`pytorch` (gpu)| `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch` |


5. To open the notbooks, open Ananconda Navigator or run `jupyter notebook` in the terminal (or `Anaconda Prompt` in Windows) while the `deep_learn` environment is activated.
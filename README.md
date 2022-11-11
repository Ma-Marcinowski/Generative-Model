# Generative-Model

### 0. Introduction

* #### 0.1. Objective of the author's repository was to introduce GAN based methods for forgery of static handwritten signatures. The purpose of these methods was to evaluate and develop forensic methods for detection of deepfake signatures.

* #### 0.2. This repository presents the final version of the model.

* #### 0.3. Keywords:

    * Computational, statistical, probabilistic; 
    * Forensics, criminalistics, analysis, examination;
    * Handwriting, signatures, documents, forgery;
    * Neural, networks, deep, machine, learning, artificial, intelligence, generative, adversarial, GAN, ANN, AI.
    
* #### 0.4. Implementation
   
  * Programming language: Python 3.
   
  * API: TensorFlow Core v2.6.0 (stable).
   
  * Execution: Google Colaboratory.
    
### 1. Data

* #### 1.1. 
  
* #### 1.2. Training and validation subsets:
    
    *

### 2. Preprocessing

* #### 2.1. General preprocessing (code available at /Preprocessing/Preprocessing.py):
    
    *

* #### 2.2. Target data (code available at /Preprocessing/Preprocessing.py):
    
    *

* #### 2.3. Input data (code available at /Preprocessing/Preprocessing.py):
    
    *

* #### 2.4. Mask data (code available at /Preprocessing/Averaging.py):
    
    *

* #### 2.5. Validation data (code available at /Preprocessing/Augmenting.py):
    
    *

### 3. Dataframing (code available at /Dataframing/Dataframing.py):

* ####

### 4. Model

* #### 4.1. Based on:
     
     * Isola, Phillip, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. ‘Image-to-Image Translation with Conditional Adversarial Networks’. ArXiv:1611.07004 [Cs], 26 November 2018. http://arxiv.org/abs/1611.07004.

     * Park, Taesung, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. ‘Semantic Image Synthesis with Spatially-Adaptive Normalization’. arXiv, 5 November 2019. https://doi.org/10.48550/arXiv.1903.07291.
     
* #### 4.2. Architecture (code available at /Model/GAN_Model.py):

     *

* #### 4.3. Hyperparameteres:

    * Loss: Wasserstein Hinge Loss (based on: The TensorFlow GAN Authors, Losses that are useful for training GANs, https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py)
    
    * Optimizer - Adam (Adaptive Moment Estimation):
    
       * Parameters as recommended by: Isola, Phillip, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. ‘Image-to-Image Translation with Conditional Adversarial Networks’. ArXiv:1611.07004 [Cs], 26 November 2018. http://arxiv.org/abs/1611.07004.
      
    * Batchsize - 1.
    
    * Generator / discriminator update rates - 1.

* #### 4.4. Training (training log is available at /Training/Training_Log.csv):

    *
    
     ![training_loss](https://github.com/Ma-Marcinowski/Generative-Model/blob/main/Training/Training_Loss.png "Training_Loss")


* #### 4.5. Validating (validation results are available at /Validationg/Results.png):

     
     ![results](https://github.com/Ma-Marcinowski/Generative-Model/blob/main/Validating/Results.png "Results")
     

### 6. Discrite cosine transform for deepfake detection (code available at /DCT/DCT.py)

* #### 6.1. Based on the paper: Frank, Joel, Thorsten Eisenhofer, Lea Schönherr, Asja Fischer, Dorothea Kolossa, and Thorsten Holz. ‘Leveraging Frequency Analysis for Deep Fake Image Recognition’. arXiv, 26 June 2020. https://doi.org/10.48550/arXiv.2003.08685.

* #### 6.2. Discrite cosine transform (DCT) may be utilized to detect deepfakes, however, the author found no artifacts that could undoubtedly prove the generated signatures as fake. 

* #### 6.3. Spectra were averaged over validation results (Fake Signatures) and corresponding true signatures (True Signatures).

     ![spectra](https://github.com/Ma-Marcinowski/Generative-Model/blob/main/DCT/Spectra.png "Spectra")

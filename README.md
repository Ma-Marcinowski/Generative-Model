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

* #### 1.1. CEDAR database of 2640 scans of static anonymous signatures (1320 true and 1320 fake signatures):

    * M.K. Kalera, S. Srihari, A. Xu, Offline signature verification and identification using distance statistics, „International Journal of Pattern Recognition and Artificial Intelligence” t. 18 nr 07 (2004), DOI: 10.1142/S0218001404003630.
     
    * S. Dey, A. Dutta, J.I. Toledo, S.K. Ghosh, J. Llados, U. Pal, SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification, „arXiv:1707.02131 [cs]” (2017), http://arxiv.org/abs/1707.02131
  
* #### 1.2. Utlized:

    * 240 scans of true static signatures by 10 writers (24 per writer).
    * 10 randomly augmented signatures were picked for validation, one per writer (along with 10 corresponding not-augmented signatures).

### 2. Preprocessing

* #### 2.1. General preprocessing (code available at /Preprocessing/Preprocessing.py):
    
    * Grayscale;
    * Colour inversion;
    * Zero padding to achieve equal height and width;
    * Resizing to 512x512px;
    * Denoiseing by thresholding of pixel values below 25 to zero. 

* #### 2.2. Training input data (code available at /Preprocessing/Preprocessing.py):
    
    * Vide 2.1.
    * Gaussian blurr;
    * Canny edge detection.

* #### 2.3. Validation input data (code available at /Preprocessing/Augmenting.py):
    
    * Vide 2.2.
    * Augmented with randomized elastic deformations (https://pypi.org/project/elasticdeform/);
    * Augmented by random removal of 128x128px areas from images.
    
* #### 2.4. Target data (code available at /Preprocessing/Preprocessing.py):
    
    * Vide 2.1.
    
* #### 2.5. Mask data (code available at /Preprocessing/Averaging.py):
    
    * Vide 2.1.
    * Average of preprocessed/target images (separately for each writer).
    
### 3. Dataframing (code available at /Dataframing/Dataframing.py):

    * During training, each writer is represented by his/hers true signatures and their corresponding edges (target and input data), and by his/hers averaged signatures (mask data).
    * During validation, there are only two target-input pairs per writer, both of the same signature, but one with augmented input.
    * During testing, all input signatures are augmented.

### 4. Model

* #### 4.1. Based on:
     
     * Isola, Phillip, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. ‘Image-to-Image Translation with Conditional Adversarial Networks’. ArXiv:1611.07004 [Cs], 26 November 2018. http://arxiv.org/abs/1611.07004.

     * Park, Taesung, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. ‘Semantic Image Synthesis with Spatially-Adaptive Normalization’. arXiv, 5 November 2019. https://doi.org/10.48550/arXiv.1903.07291.
     
* #### 4.2. Architecture (code available at /Model/GAN_Model.py):

     * Because we are processing 512x512px images, hence there are two additinal layers in the generator (one for the encoder and one for the decoder).
     
     * Instead of batch-normalization, generator utilizes SPADE layers, and the discriminator utilizes instance-normalization.
     
     * Generator takes as input edges of signatures, and as semantic masks (vide SPADE) it takes writer-wise averages of signatures.
     
     * Discriminator takes as input edges of signatures and either targer (i.e. true) signatures or the ones produced by the generator.

* #### 4.3. Hyperparameteres:

    * Loss: Wasserstein Hinge Loss (based on: The TensorFlow GAN Authors, Losses that are useful for training GANs, https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py)
    
    * Optimizer - Adam (Adaptive Moment Estimation):
    
       * Parameters as recommended by: Isola, Phillip, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. ‘Image-to-Image Translation with Conditional Adversarial Networks’. ArXiv:1611.07004 [Cs], 26 November 2018. http://arxiv.org/abs/1611.07004.
      
    * Batchsize - 1.
    
    * Generator / discriminator update rates - 1.

* #### 4.4. Training (training log is available at /Training/Training_Log.csv):

    * Convergent, discontinued after 260 epochs because of minimal gains in the image quality.
    
     ![training_loss](https://github.com/Ma-Marcinowski/Generative-Model/blob/main/Training/Training_Loss.png "Training_Loss")


* #### 4.5. Validating (validation results are available at /Validationg/Results.png):

     
     ![results](https://github.com/Ma-Marcinowski/Generative-Model/blob/main/Validating/Results.png "Results")
     

### 6. Discrite cosine transform for deepfake detection (code available at /DCT/DCT.py)

    * Based on the paper: Frank, Joel, Thorsten Eisenhofer, Lea Schönherr, Asja Fischer, Dorothea Kolossa, and Thorsten Holz. ‘Leveraging Frequency Analysis for Deep Fake Image Recognition’. arXiv, 26 June 2020. https://doi.org/10.48550/arXiv.2003.08685.

    * Discrite cosine transform (DCT) may be utilized to detect deepfakes, however, the author found no artifacts that could undoubtedly prove the generated signatures as fake. 
    
    * Spectra were averaged over validation results (Fake Signatures) and corresponding true signatures (True Signatures).

     ![spectra](https://github.com/Ma-Marcinowski/Generative-Model/blob/main/DCT/Spectra.png "Spectra")

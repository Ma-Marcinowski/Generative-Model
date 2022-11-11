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

* #### 2.1. General preprocessing (code available at /Preprocessing.py):
    
    *

* #### 2.2. Target data (code available at /Preprocessing.py):
    
    *

* #### 2.3. Input data (code available at /Preprocessing.py):
    
    *

* #### 2.4. Mask data (code available at /Averaging.py):
    
    *

* #### 2.5. Validation data (code available at /Augmenting.py):
    
    *

### 3. Dataframing

* ####

### 4. Model

* #### 4.1. Based on:
     
     * Isola, Phillip, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. ‘Image-to-Image Translation with Conditional Adversarial Networks’. ArXiv:1611.07004 [Cs], 26 November 2018. http://arxiv.org/abs/1611.07004.

     * Park, Taesung, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. ‘Semantic Image Synthesis with Spatially-Adaptive Normalization’. arXiv, 5 November 2019. https://doi.org/10.48550/arXiv.1903.07291.
     
* #### 4.2. Architecture (code available at /GAN_Model.py):

      *

* #### 4.3. Training:

      *

### 5. Results

     ![results](https://github.com/Ma-Marcinowski//Generative-Model/blob/main/Results.png "Results")

### 6. Discrite cosine transform for deepfake detection (code available at /DCT.py)

* #### 6.1. Based on the paper: Frank, Joel, Thorsten Eisenhofer, Lea Schönherr, Asja Fischer, Dorothea Kolossa, and Thorsten Holz. ‘Leveraging Frequency Analysis for Deep Fake Image Recognition’. arXiv, 26 June 2020. https://doi.org/10.48550/arXiv.2003.08685.

* #### 6.2. Discrite cosine transform (DCT) may be utilized to detect deepfakes, however, the author found no artifacts that could undoubtedly prove the generated signatures as fake. 

* #### 6.3. Results were averaged over validation results and corresponding true signatures.

     ![dct](https://github.com/Ma-Marcinowski//Generative-Model/blob/main/DCT.png "DCT")

## <a href="http://www.insa-toulouse.fr/" ><img src="http://www.math.univ-toulouse.fr/~besse/Wikistat/Images/Logo_INSAvilletoulouse-RVB.png" style="float:left; max-width: 80px; display: inline" alt="INSA"/> |  [*Mathématiques Appliquées*](http://www.math.insa-toulouse.fr/fr/index.html), [`Science des Données`](http://www.math.insa-toulouse.fr/fr/enseignement.html) 

# High Dimensional and Deep Learning

## Presentation :

 

The main theme of the course is learning methods, especially deep neural networks, for  processing  high dimensional  data, such as signals or images. We will cover the following topics:

 
* Neural networks and introduction to deep learning: definition of neural networks, activation functions, multilayer perceptron, backpropagation algorithms, optimization algorithms, regularization  
**Application** : [Implementation of a mlp with one layer with `Numpy`](https://github.com/wikistat/High-Dimensional-Deep-Learning/tree/master/BackPropagation)


* Convolutional neural networks: convolutional layer, pooling, dropout, convolutional network architectures (ResNet, Inception), transfer learning and fine tuning, applications for image or signal classification.  
**Application** : [Image classification on *MNIST* and *CatsVsDogs* data with `Tensorflow`](https://github.com/wikistat/High-Dimensional-Deep-Learning/tree/master/ImageClassification)


* Encoder-decoder, Variational auto-encoder, Generative adversarial networks

* Functional decomposition on splines, Fourier or wavelets bases: cubic splines, penalized least squares criterion, Fourier basis, wavelet bases, applications to nonparametric regression, linear estimators and nonlinear estimators by thresholding, links with the LASSO method.

* Anomaly detection for functional data: One Class SVM, Random Forest, Isolation Forest, Local Outlier Factor. Applications to  anomaly detection in functional data.
 
* Deep learning for time series forecasting
 
* Object detection / image segmentation

------------
 

## Organisation : 

* Lectures : 15 H .

* Practical works : 28 H applications on real data sets with the softwares R and Python's libraries Scikit Learn and Keras -Tensorflow. 

## Evaluation

* written exam (50 %) - 14/01/2021

* project (oral presentation 25% - 18/01/2021 + notebook (25%) <br>The main of this project is to apply the knowledge you acquired during this course by:

    * Selecting a deep learning algorithm you haven't seen in this course.
    * Explaining how this algorithm works (oral presentation).
    * Apply these algorithm on a dataset and explain their performances (notebook and oral presentation).

You can choose a deep learning algorithm among the following list. <br>
This list is not exhaustive and you can suggest other algorithms (that's actually a good idea). <br>
Also, the code proposed on those examples are not necessarily the official code nor the one propose by the authors. <br>


**Example of algorithms**
* Detection & segmentation
    * Yolo [paper](https://arxiv.org/abs/1804.02767), [code](https://github.com/pjreddie/darknet)
    * U-net [paper](https://arxiv.org/abs/1505.04597), [code](https://www.tensorflow.org/tutorials/images/segmentation)
    * Mask Rcnn, Fast Rcnn, Faster Rcnn [paper](https://arxiv.org/pdf/1506.01497.pdf), [code](https://github.com/rbgirshick/py-faster-rcnn)
* One shot learning
    * Siamese Network [paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf), [code](https://github.com/tensorfreitas/Siamese-Networks-for-One-Shot-Learning)
* Outlier detection
    * Deep roc
* Style Transfer 
    * A Neural Algorithm of Artistic Style [paper](https://arxiv.org/abs/1508.06576), [code](https://www.tensorflow.org/tutorials/generative/style_transfer)
    * Cycle gan [paper](https://arxiv.org/pdf/1703.10593.pdf), [code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* Generative model
    * Pixel Cnn/++ [paper](https://arxiv.org/abs/1606.05328), [code](https://github.com/openai/pixel-cnn)
    * Gan variation [paper](https://arxiv.org/abs/1701.07875), [code](https://github.com/martinarjovsky/WassersteinGAN)
    
* Anomaly detection 
   * Deep robust One -Class classification [paper](https://arxiv.org/pdf/2002.12718.pdf), [code](https://github.com/microsoft/EdgeML)
    
    





   
   


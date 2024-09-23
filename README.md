# Image-Captioning-using-CNNs-LSTMs
Image Captioning is the process of generating textual description of an image. It uses both Natural
Language Processing and Computer Vision to generate the captions.<br/>
To perform Image Captioning we will require two deep learning models <br/>
## CNN (Convolutional Neural Network) 
Extract the features from the image of some vector
size aka the vector embeddings. The size of these embeddings depend on the type of
pretrained network being used for the feature extraction <br/>
## Long Short-Term Memory (LSTM) 
LSTM are equipped with memory cells and gating mechanisms, allowing
them to efectively address the vanishing gradient problem and capture temporal dependencies.
LSTMs are used for the text generation process. The image embeddings are concatenated with the word
embeddings and passed to the LSTM to generate the next word <br/>
## Data and Pre-processing:
Dataset link https://www.kaggle.com/datasets/adityajn105/flickr8k <br/>
The Flickr8k dataset is utilized, and pre-processing involves caption normalization, and dataset splitting. Image processing includes conversion to arrays, and feature extraction using a pre-trained Inception v3 model are applied. The dataset is prepared using generators for optimized memory usage during training.<br/>
## InceptionV3 model
inception_v3_model.load_weights('/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5') <br/>
Keras Inception V3 model weights link https://www.kaggle.com/datasets/madmaxliu/inceptionv3 <br/>
## Defining the Image Captioning Model
### Encoder (image features)
Input Layer <br/>
Batch Normalization Layer<br/>
Dense Layer<br/>
Batch Normalization Layer<br/>
### Decoder (captions)
Input Layer<br/>
Embedding Layer<br/>
LSTM Layer<br/>
### Output
Add Layer (Encoder output + Decoder output)<br/>
Dense Layer + ReLU activation function<br/>
Dense Layer + Softmax activation function <br/>
## Inference
### Caption Generation:
At each time step, the model takes as input the image features along with the generated words (starting with start at the first time step), predicting the probabilities of the next word.<br/>
### Greedy algorithm:
To select the best caption, the greedy algorithm is employed. This method, chooses the most probable word at each time step and appends it to the generated captions until the selected word is the end token, or the length of the decoded captions exceeds the maximum sequence length.<br/>
### Beam Search:
Beam Search is an alternative to the greedy algorithm for selecting captions. It maintains a beam of multiple hypotheses (candidate captions) at each time step. At each step, the model predicts the next word for each hypothesis. The top-k candidates (based on probabilities) are retained in the beam. The process continues until the end token is generated or the maximum sequence length is reached.<br/>
Evaluation is performed using the BLEU score<br/>
Visualization: The function visualization() plots the images along with their corresponding predicted captions, accompanied by 2 BLEU scores.<br/>

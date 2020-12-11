# Real-time Accident Detection On IoT Device

<b>An Accident Detector on IOT devices</b>


<h4>Table of Contents</h4>

[Summary](#Summary)<br /> 
[Dataset Generation](#Data)<br />
[Data Processing](#Data_processing)<br />
[Model Training](#Model)<br />
[Pre-trained Model](#Pre-trained)<br />
[References](#References)<br />

<a name="Summary"/>
<h3>Summary</h3>

The model implements CNN(Convolutional Neural Network)-based learning on accident footages captured through traffic cameras. The model aims to detect Real-time Accident via an IoT-based traffic camera.
<br /><br />

<a name="Data"/>
<h3>Dataset Generation</h3>

For the dataset generation, We use the CADP Dataset which comprises a collection of accident videos and DETRAC dataset containing footages of normal traffic flow.
Initially, we supposedly marked the 'CADP' as an Accident Dataset and 'DETRAC' as a non-accident set. Since the CNN model ought be applied on images, rather than videos, we converted the video footages into images(using frame_cutter.py). Training on this 'Crude Dataset', the initial training and validation error was significant, partly due to the large number of 'False Negative' in the 'Accident Dataset'. In order to overcome this, we manually removed the false negatives from 'Accident Dataset'. Finally, we obtained the final 'Compact Dataset' with 4000 accident images from 'Accident Dataset' and 8000 images from 'Non-accident Dataset' (12000 images overall). The 'Compact Dataset' can be found here <a href="https://drive.google.com/drive/folders/1oR_e3g257MnhEOiNJbk3lLoPmkxSWNE8?usp=sharing">Compact Dataset</a>. In the next section, we describe the Data Preprocessing employed before training the model.
<br /><br />

<a name="Data_processing"/>
<h3>Data Processing</h3>

We converted each video frame into an image. Each of these images is a two-dimensional array of pixels, where each pixel has information about the red, green, and blue (RGB) color levels. To reduce the dimensionality at the individual image level, we convert the 3-D RGB color arrays to grayscale. Additionally, to make the computations more tractable on a CPU, we resize each image to (128, 128) - in effect reducing the size of each image to a 2-D array of 122 X 128.
<br /><br />

<a name="Model"/>
<h3>Model Training</h3>

We built a convolutional neural network for image classification with keras.

We created a sequential model which linearly stacks all the the layers, using keras.models. We implemented different keras layers like Conv2D- convolutional layer, MaxPooling2D- max pooling layer, Dropout, Dense- add neurons, Flatten- convert the output to 1D vector, using keras.layers along with 'ReLU' and 'softmax' as activation functions.

While compiling the model we used "sparse_categorical_crossentropy" as our loss function, "adam" as our optimizer and "accuracy metrics" as metrics. Then in model fit we ran the model for "10" epochs with "0.2" validation_split.

Using this model we are able to predict whether given video contains accident or not.

### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.

##### pip install requirements.txt

<h3>Procedure</h3>

1) Run `create_dataset.py` for converting the video to images.
2) Then run `main.py` to train the model.
3) Finally, run `model.py` for testing your model.

<a name="Pre-trained"/>
<h3>Pre-trained Model</h3>

The Pre-trained Model can be found here <a href="https://drive.google.com/drive/folders/133RyXB-OSqB7YozcSb8FTYB8bBisjOPj?usp=sharing">Pre-trained Model</a>.
Extract the zip file into the 'master folder' of your model. And, finally run 'test.py' to obtaining accuracy on 'test set'.

### Contributors

##### 1) [Sahma Anwar](https://github.com/Sahma61)
##### 2) [Onkar Telange](https://github.com/om1621)
##### 3) [Shreyansh Sahu](https://github.com/23nobody)

<a name="References"/>
<h3>References</h3>

<ul>
<li> <a href="https://ankitshah009.github.io/accident_forecasting_traffic_camera">CADP Dataset</a>
<li> <a href="http://detrac-db.rit.albany.edu/">DETRAC Dataset</a>
<li> <a href="https://drive.google.com/drive/folders/1oR_e3g257MnhEOiNJbk3lLoPmkxSWNE8?usp=sharing">Manually Created Dataset</a>

</ul>


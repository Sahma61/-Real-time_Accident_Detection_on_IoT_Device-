# Real-time Accident Detection On IoT Device

<b>An Accident Detector on IOT devices</b>


<h4>Table of Contents</h4>

[Summary](#Summary)<br /> 
[Dataset Creation](#Data)<br />
[Data Processing](#Data_processing)<br />
[The Algorithm](#Model)<br />
[References](#References)<br />

<a name="Summary"/>
<h3>Summary</h3>

The Convolutional neural network algorithm model has created to detect accidents in video footages captured by IOT devices
<br /><br />

<a name="Data"/>
<h3>Dataset Creation</h3>

For the dataset creation purpose we used the CADP Dataset which conatains the accident videos and DETRAC dataset which is mainly used for the vehicular detection 
as our non-accident videos. When we converted the CADP videos into frames (using frame_cutter.py), we found that most of the frames like frames before accident have no accident visible in them so after converting all the videos in frames we manually separated the frames with accidents. By doing this we were able to get 4000 images with accidents and similary, we got 8000 images from DETRAC dataset. So we created a dataset containing 12000 accident and non-accident images. We have uploaded the dataset to google drive and link for the same is given below.
<br /><br />

<a name="Data_processing"/>
<h3>Data Processing</h3>

We converted each video into its individual frames. Each of these images is a two-dimensional array of pixels where each pixel has information about the red, green, and blue (RGB) color levels. To reduce the dimensionality at the individual image level, we convert the 3-D RGB color arrays to grayscale. Additionally, to make the computations more tractable on a CPU, we resize each image to (128, 128) - in effect reducing the size of each image to a 2-D array of 122 X 128.
<br /><br />

<a name="Model"/>
<h3>The Algorithm</h3>

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


### Contributors

##### 1) [Sahma Anwar](https://github.com/Sahma61)
##### 2) [Onkar Telange](https://github.com/om1621)
##### 3) [Shreyansh Sahu](https://github.com/23nobody)
##### 4) [Rahul Thalor](https://github.com/RahulThalor)

<a name="References"/>
<h3>References</h3>

<ul>
<li> <a href="https://ankitshah009.github.io/accident_forecasting_traffic_camera">CADP Dataset</a>
<li> <a href="http://detrac-db.rit.albany.edu/">DETRAC Dataset</a>
<li> <a href="#">Manually Created Dataset</a>

</ul>


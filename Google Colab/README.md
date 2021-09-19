Distorted Number Recognition


**Program and Goal**
The aim of this project is to compare the recognition accuracy of distorted numbers taken from the MNIST dataset using a trained CNN model. The same model architecture was trained on the standard MNIST dataset without image distortion. The accuracy of 59% was achieved for the Distorted Model as opposed to the 98% of the Standard Model. The random distortion range applied was: Shear = up to 30%, Rotation = up to 180deg, Width = up to 30% and Height = up to 30%. 


**The Data**
The data used in this program was the MNIST dataset loaded from tensorflow at tensorflow.keras.datasets.


**Data Overview**
The MNIST dataset comes with a set of 70,000 images of handwritten numbers normalised to 28 pixels by 28 pixels. The dataset is split into 60,000 training images and 10,000 test images on load. For more details about the dataset, you may visit https://en.wikipedia.org/wiki/MNIST_database.


**Structure & Approach**
Two different datasets were modelled for comparison. One model used the MNIST dataset as-is while the other was modelled with the same dataset but with random transformation introduced. The transformation done on the latter was to simulate real-life recognition of documents whereby there maybe inherent noise from the source. The models were labelled "Standard" and "Distorted" respectively.

Both models were fitted with the same neural network configuration so that their performance can be compared. The distortions introduced to the "Distorted" model were applied using the ImageDataGenerator function from tensorflow.keras.preprocessing with the following configuration arbituarily chosen:

- 180 Degrees rotation
- 30% Width and height shifts
- 30% Shear
- Filling void pixels with the nearest one.

The program is divided into the following 5 sections:

Section 1 : Explore Data 
Section 2 : Data Preprocessing
Section 3 : Model Training and Fitting
Section 4 : Model Evaluation
Section 5 : Results and Analysis 


**Result Analysis**
Standard Model
As expected, model produced good results with accuracy, precision and recall rates of more than 95%. Model also correctly predicted the test prediction, although strictly speaking, this data used for the last prediction should be "new" and not extracted from the test set. Also expected was that this model performs much better than the Distorted Number Model although the Distorted Number Model may be more suitable for real-life scenarios.

Distorted Model
Model had a comparatively low accuracy of 59% (vs 98%) when matched against the Standard Number Model. The recall rates for number 1 and 0 were high. It was likely due to the fact that the general shape did not change much given any transformation of the image. Same reason might also had attributed to the high precision rate of number 1 and 0. The confusion showed by the heatmap on the confusion matrix for numbers 1 and 7, 6 and 9 and 3 and 5 could be explained by the similarities between them when they were rotated. In the last prediction, the model predicted the image of 2 to be a number 4. This was a good example which shows how transformation combination could make a number look like another. To a human eye, the image looked more like a flipped 4 than the actual 2 given some trimmed off parts of the image. In summary, although a Standard Number Model provides a good training resource, fitting a model to real-life documents with inherent noise might not be as straightforward.



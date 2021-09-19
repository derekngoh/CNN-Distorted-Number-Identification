import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report
import random, argparse

from CNN_Model import CNN_Model

cmdparser = argparse.ArgumentParser(description='distorted model')
cmdparser.add_argument('--filters', help='set the number of filters', default='64')
cmdparser.add_argument('--patience', help='set number of patience for earlystop', default='10')
cmdparser.add_argument('--training_epochs', help='number of epochs to train for', default='100')
cmdparser.add_argument('--distortion_params', help='set parameters for distortion. Rotation in degrees, width between 0 and 1, height between 0 and 1, shear between 0 and 1] e.g. 180 0.3 0.3 0.3]', nargs="*", type=float, default=[180,0.3,0.3,0.3])
args = cmdparser.parse_args()

filters = int(args.filters)
patience = int(args.patience)
training_epochs = int(args.training_epochs)
distortion_params = args.distortion_params

#PREPROCESS mnist dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()

distt_model = CNN_Model()
distt_model.set_xy_original_data(x_train,y_train,x_test,y_test,y_to_cat=True)

#CREATE generator to distort images
distt_model.create_distortion_generator(distortion_params)

#RESHAPE data for model fitting
distt_model.xtrain = distt_model.xtrain.reshape(60000,28,28,1)
distt_model.xtest = distt_model.xtest.reshape(10000,28,28,1)

#CREATE distorted image for fitting
distt_model.distort_image_data()

random_num = random.randint(1,999)

#COMPILE, fit and save model with desired configuration
num_of_filters = filters
early_stop_patience = patience #0 for no earlystop
epochs = training_epochs

distt_model.create_standard_model(num_of_filters)
distt_model.fit_model_with_validation(epochs,early_stop_patience,generator=True)
distt_model.main_model.save(distt_model.set_current_path("distorted_model{a}.h5".format(a=random_num)))
distt_model.save_loss_plot("distorted_model_loss_plot{a}.jpg".format(a=random_num), show=True)

#USE the model to predict the test data.
predictions = np.argmax(distt_model.main_model.predict(distt_model.distorted_xtest_gen),axis=-1)
results = classification_report(np.argmax(distt_model.ytest,axis=1),predictions)
sns.heatmap(confusion_matrix(np.argmax(distt_model.ytest,axis=1),predictions))
plt.savefig(distt_model.set_current_path("distorted_model_heatmap{a}".format(a=random_num)))
distt_model.save_text(results,"distorted_model_results{a}.txt".format(a=random_num))

print(results)
plt.show()

#PICK randomly chosen image, though strictly speaking, we should have used totally new data.
index = random.randint(0,9999)
sample_image = distt_model.xtest[index].reshape(28,28,1)
image_number = np.argmax(distt_model.ytest[index])
distorted_image = distt_model.distortion_gen.random_transform(sample_image)
distorted_image = distorted_image.reshape(28,28)
plt.imshow(distorted_image, cmap='Greys')
plt.show()

#PREDICT on 'new' data
scaled_distorted = distorted_image/255
scaled_distorted = scaled_distorted.reshape(1,28,28,1)
pred = np.argmax(distt_model.main_model.predict(scaled_distorted),axis=-1)
pred_text = "The image displays the number '{a}' and the model predicted the image to be showing {b}.".format(a=image_number,b=pred)
distt_model.save_text(pred_text,"distorted_model_results{a}.txt".format(a=random_num))
print(pred_text)
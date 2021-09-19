import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report
import random, argparse

from CNN_Model import CNN_Model

cmdparser = argparse.ArgumentParser(description='standard model')
cmdparser.add_argument('--filters', help='set the number of filters', default='64')
cmdparser.add_argument('--patience', help='set number of patience for earlystop', default='10')
cmdparser.add_argument('--training_epochs', help='number of epochs to train for', default='100')
args = cmdparser.parse_args()

filters = int(args.filters)
patience = int(args.patience)
training_epochs = int(args.training_epochs)

#PREPROCESS mnist dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()
std_model = CNN_Model()
std_model.set_xy_original_data(x_train,y_train,x_test,y_test,y_to_cat=True)
std_model.xtrain = std_model.xtrain.reshape(60000,28,28,1)
std_model.xtest = std_model.xtest.reshape(10000,28,28,1)
std_model.scale_xtrain_mnist()

random_num = random.randint(1,999)

#COMPILE, fit and save model
num_of_filters = filters
early_stop_patience = patience #0 for no earlystop
epochs = training_epochs

std_model.create_standard_model(num_of_filters)
std_model.fit_model_with_validation(epochs,early_stop_patience)
std_model.main_model.save(std_model.set_current_path("std_model{a}.h5".format(a=random_num)))
std_model.save_loss_plot("std_model_loss_plot{a}.jpg".format(a=random_num), show=True)

#PREDICT with trained model on the validation set
predictions = np.argmax(std_model.main_model.predict(std_model.scaled_xtest),axis=-1)
results = classification_report(np.argmax(std_model.ytest,axis=1),predictions)
sns.heatmap(confusion_matrix(np.argmax(std_model.ytest,axis=1),predictions))
plt.savefig(std_model.set_current_path("std_model_heatmap{a}".format(a=random_num)))
std_model.save_text(results,"std_model_results{a}.txt".format(a=random_num))

print(results)
plt.show()

#Pick randomly chosen image, though strictly speaking, we should have used totally new data.
index = random.randint(0,9999)
sample_image = std_model.xtest[index].reshape(28,28,1)
image_number = np.argmax(std_model.ytest[index])
sample_image = sample_image.reshape(28,28)
plt.imshow(sample_image, cmap='Greys')
plt.show()

#PREDICT on 'new' data
pred = np.argmax(std_model.main_model.predict(sample_image.reshape(1,28,28,1)))
pred_text = "The image displays the number '{a}' and the model predicted the image to be showing {b}.".format(a=image_number,b=pred)
std_model.save_text(pred_text,"std_model_results{a}.txt".format(a=random_num))
print(pred_text)
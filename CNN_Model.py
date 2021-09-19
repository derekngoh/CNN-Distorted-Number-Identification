import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import Utils

class CNN_Model():
    def __init__(self) -> None:
        self.main_model = None
        self.scaler = None
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None
        self.scaled_xtrain = None
        self.scaled_xtest = None
        self.distortion_gen = None
        self.distorted_xtrain_gen = None
        self.distorted_xtest_gen = None

    def set_xy_original_data(self,xtrain,ytrain,xtest,ytest,y_to_cat=False):
        self.xtrain = np.array(xtrain)
        self.ytrain = np.array(ytrain)
        self.xtest = np.array(xtest)
        self.ytest = np.array(ytest)
        if (y_to_cat): 
            self.ytrain = np.array(to_categorical(ytrain))
            self.ytest = np.array(to_categorical(ytest))

    def scale_xtrain_mnist(self):
        self.scaled_xtrain = self.xtrain/255
        self.scaled_xtest = self.xtest/255

    def create_standard_model(self, num_of_filters=64):
        model = Sequential()
        model.add(Conv2D(filters=num_of_filters,kernel_size=(2,2),input_shape=(28,28,1),
                        padding='valid',activation='relu',))
        model.add(MaxPool2D(pool_size=(4,4)))

        model.add(Conv2D(filters=num_of_filters/2,kernel_size=(2,2),
                        padding='valid',activation='relu',))
        model.add(MaxPool2D(pool_size=(2,2)))

        model.add(Flatten())

        model.add(Dense(num_of_filters*2,activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(num_of_filters,activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(10,activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.main_model = model
        return self.main_model.summary()

    def fit_model_with_validation(self,epochs,patience,generator=False):
        early_stop = EarlyStopping(monitor='val_loss',patience=patience)
        if (generator):
            self.main_model.fit_generator(self.distorted_xtrain_gen,
                                epochs=epochs,
                                callbacks=[early_stop],
                                validation_data=self.distorted_xtest_gen)
        else:
            self.main_model.fit(self.scaled_xtrain,
                    self.ytrain,
                    epochs=epochs,
                    callbacks=[early_stop],
                    validation_data=(self.scaled_xtest,self.ytest))

    def create_distortion_generator(self, distortion_params=[180,0.3,0.3,0.3]):
        params = distortion_params
        rotation = int(params[0])
        width = float(params[1])
        height = float(params[2])
        shear = float(params[3])
        self.distortion_gen = ImageDataGenerator(rotation_range=rotation, 
                               width_shift_range=width,
                               height_shift_range=height,
                               shear_range=shear,
                               rescale=1/255,
                               fill_mode='nearest')

    def distort_image_data(self):
        self.distorted_xtrain_gen = self.distortion_gen.flow(self.xtrain,self.ytrain)
        self.distorted_xtest_gen = self.distortion_gen.flow(self.xtest,self.ytest,shuffle=False)

    def save_loss_plot(self,filename,show=False,figsize=(15,6)):
        Utils.save_show_loss_plot(self.main_model,filename,show=show,figsize=figsize)

    def set_current_path(self,filename=None):
        return Utils.get_set_current_path(filename)

    def save_text(self,content, filename):
        Utils.save_as_text_file(content,filename)

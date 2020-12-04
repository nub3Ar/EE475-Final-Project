import keras
from layers import DropConnect

INPUT_SHAPE = (28,28,1)
NUM_CLASSES = 10

class CNN(keras.Model):

    def __init__(self):
        super(CNN, self).__init__()
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='Same', activation='relu',
                                      input_shape = INPUT_SHAPE) )
        self.model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))
        self.model.add(keras.layers.Conv2D(filters=64,kernel_size=(3, 3),strides=1,padding='same', activation='relu'))
        self.model.add(keras.layers.MaxPool2D(pool_size=(2, 2),padding='same'))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(units=NUM_CLASSES, activation='sigmoid'))

    def call(self, inputs):
        return self.model(inputs)

class CNN_Drop(keras.Model):

    def __init__(self):
        super(CNN_Drop, self).__init__()
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='Same', activation='relu',
                                      input_shape = INPUT_SHAPE) )
        self.model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))
        self.model.add(keras.layers.Conv2D(filters=64,kernel_size=(3, 3),strides=1,padding='same', activation='relu'))
        self.model.add(keras.layers.MaxPool2D(pool_size=(2, 2),padding='same'))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(units=NUM_CLASSES, activation='sigmoid'))

    def call(self, inputs):
        return self.model(inputs)

class CNN_BN(keras.Model):

    def __init__(self):
        super(CNN_BN, self).__init__()
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='Same', input_shape = INPUT_SHAPE) )
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.ReLU())
        self.model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))
        
        self.model.add(keras.layers.Conv2D(filters=64,kernel_size=(3, 3),strides=1,padding='same'))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.ReLU())
        self.model.add(keras.layers.MaxPool2D(pool_size=(2, 2),padding='same'))
        
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(units=NUM_CLASSES, activation='sigmoid'))

    def call(self, inputs):
        return self.model(inputs)

if __name__ == '__main__':
    print('model import working')

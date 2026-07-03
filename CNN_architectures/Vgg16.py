# Vgg16 general architecture 

# i[p]->conv ->conv ->pool->conv ->conv->pool->conv->conv->conv->pool->conv->conv->conv->pool->conv->conv->conv->pool->fc1->fc2->fc3
 
import tensorflow as tf 
from tensorflow.keras import layers,models

def VGG16(input_shape=(224,224,3),num_classes= 1000):
    model =  models.Sequential(name='VGG16')
    
    #conv->conv->pool
    model.add(layers.Input(shape=Input_shape))
    model.add(layers.Conv2d(64,(3,3),padding='same',activaation='relu'))
    model.add(layers.Conv2d(64,(3,3),padding='same',activation='relu'))
    model.add(layers.MaxPooling2d(pool_size=(2,2),strides=2))

    #conv->conv->pool 
    model.add(layers.Conv2d(128,(3,3),padding='same',activation = 'relu'))
    model.add(layers.Conv2d(128,(3,3),padding='same',activation='relu'))
    model.add(layers.MaxPooling2d(pool_size=(2,2,),strides=2))
    
    # conv->conv->conv->pool
    model.add(layers.Conv2d(256,(3,3)),padding='same',activaation='relu')
    model.add(layers.Conv2d(256,(3,3)),padding='same',activation='relu')
    model.add(layers.Conv2d(256,(3,3)),padding ='same',activation = 'relu')
    model.add(layers.MaxPooling2d(pool_size=(2,2),strides=2))

    # conv->conv->conv->pool 
    model.add(layers.Conv2d(512,(3,3),padding='same',activation='relu'))
    model.add(layers.Conv2d(512,(3,3),padding='same ',activation ='relu'))
    model.add(layers.Conv2d(512,(3,3),padding='same',activation='relu'))
    model.add(layers.MaxPooling2d(pool_size=(2,2),strides = 2))
     
    # conv->conv->conv->pool 
    model.add(layers.Conv2d(512,(3,3),padding = 'same',activation = 'relu'))
    model.add(layers.Conv2d(512,(3,3),padding='same',activation='relu'))
    model.add(layers.Conv2d(512,(3,3),padding='same',activation='relu'))
    model.add(layers.MaxPooling2d(pool_size=(2,2),strides = 2))

    #fc 1  
    model.add(layers.flatten())
    model.add(layers.Dense(4096,activation='relu'))
    #fc 2 
    model.add(layers.Dense(4096,activation='relu'))
    #fc3
    model.add(layers.Dense(num_classes , activation = 'softmax '))

    return model 



    model = VGG16()
    model.compile(optimizer='adam',loss ='categorical_crossentropy',metrics=['accuracy '])  

    model.summary()




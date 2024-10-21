import tensorflow as tf
tf.test.gpu_device_name() 
/device:GPU:0
!nvidia-smi 
Sun Apr 18 18:19:49 2021
+-------------------------------------------------------------------------
----+
| NVIDIA-SMI 460.67 Driver Version: 460.32.03 CUDA Version: 11.2
|
|-------------------------------+----------------------+------------------
----+
| GPU Name Persistence-M| Bus-Id Disp.A | Volatile Uncorr.
ECC |
| Fan Temp Perf Pwr:Usage/Cap| Memory-Usage | GPU-Util Compute
M. |
| | | MIG
M. |
|===============================+======================+==================
====|
| 0 Tesla T4 Off | 00000000:00:04.0 Off |
0 |
| N/A 56C P0 28W / 70W | 222MiB / 15109MiB | 0%
Default |
| | |
N/A |
+-------------------------------+----------------------+------------------
----+

+-------------------------------------------------------------------------
----+
| Processes:
|
| GPU GI CI PID Type Process name GPU
Memory |
| ID ID Usage
|
|=========================================================================
====|
+-------------------------------------------------------------------------
----+ 
from google.colab import drive
9 | P a g e
drive.mount('/content/drive', force_remount=True)
Mounted at /content/drive
!ls "/content/drive/My Drive"
'Colab Notebooks' Dataset
pip install Kaggle
#here we import all Libreies
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers import Conv2D,MaxPooling2D
import os
import numpy as np
import tensorflow as tf
import keras
from keras import models,layers
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Flatten,Input
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.model_selection import train_test_split 
path="/content/drive/MyDrive/Dataset/IMAGES/"
data=os.listdir(path)
data
array=[]
for index,y in enumerate(data):
 img=image.load_img(path+y,target_size=(224,224))
 img_data=image.img_to_array(img)
 array.append(img_data) 
import csv
grade={}
with open('/content/drive/MyDrive/Dataset/messidor_data.csv',newline=None)
as csvfile:
 spamreader=csv.reader(csvfile,delimiter=',',quotechar='l')
 for row in spamreader:
 if (row[3]=='0'):
 grade[row[0]]='0'
 else:
 grade[row[0]]=row[1]
label=[]
for i in data:
 label.append(grade[i]) 
 #link texttrain and test the images
train_array,test_array,y_train,y_test=train_test_split(array,label,test_si
ze=0.2,random_state=13)
label_train=np.array(y_train)
label_test=np.array(y_test)
test_array=np.array(test_array)
train_array=np.array(train_array)
np.max(train_array)
255.0
# here try to leran train_test_split
train_array=train_array/np.max(train_array)
test_array=test_array/np.max(test_array)
lb=LabelEncoder()
y_train=np_utils.to_categorical(lb.fit_transform(label_train))
y_test=np_utils.to_categorical(lb.fit_transform(label_test))
train_x,valid_x,train_label,valid_label=train_test_split(train_array,y_tra
in,test_size=0.1,random_state=13)
flat=Flatten()(Input)
x=Dense(1024,activation='relu')(flat)
x=Dropout(0.2)(x)
x=Dense(512,activation='relu')(x)
x=Dropout(0.2)(x)
x=Dense(128,activation='relu')(x)
x=Dropout(0.2)(x)
x=Dense(64,activation='relu')(x)
x=Dropout(0.2)(x)
x=Dense(32,activation='relu')(x)
prediction=Dense(5,activation="softmax")(x)
prediction
<KerasTensor: shape=(None, 5) dtype=float32 (created by layer 'dense_5')>
model=Model(inputs=Input,outputs=prediction)
model.summary() 
Model: "model"
_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
image_input (InputLayer) [(None, 224, 224, 3)] 0
_________________________________________________________________
flatten (Flatten) (None, 150528) 0
_________________________________________________________________
dense (Dense) (None, 1024) 154141696
_________________________________________________________________
dropout (Dropout) (None, 1024) 0
_________________________________________________________________
dense_1 (Dense) (None, 512) 524800
_________________________________________________________________
dropout_1 (Dropout) (None, 512) 0
_________________________________________________________________
dense_2 (Dense) (None, 128) 65664
_________________________________________________________________
dropout_2 (Dropout) (None, 128) 0
_________________________________________________________________
dense_3 (Dense) (None, 64) 8256
_________________________________________________________________
dropout_3 (Dropout) (None, 64) 0
_________________________________________________________________
dense_4 (Dense) (None, 32) 2080
_________________________________________________________________
dense_5 (Dense) (None, 5) 165
=================================================================
Total params: 154,742,661
Trainable params: 154,742,661
Non-trainable params: 0 
#adam=keras.optimizers.SGD(learning_rate=0.00001)
adam=keras.optimizers.Adam(learning_rate=0.00001)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['acc
uracy'])
cnnhistory=model.fit(train_x,train_label,batch_size=32,epochs=60,verbose=2
,validation_data=(valid_x,valid_label))
plt.plot(cnnhistory.history['loss']) 
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.legend(['train_loss','test_loss'],loc='upper right')
plt.show()
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.legend(['train_acc','test_acc'],loc='upper left')
plt.show() 
model_name='dr_cnn.h5'
save_dir=os.path.join(os.getcwd(),'save_models')
if not os.path.isdir(save_dir):
 os.makedirs(save_dir)
model_path=os.path.join(save_dir,model_name)
model.save(model_path)
print('Saved trained model at %s'%model_path)
Saved trained model at /content/save_models/dr_cnn.h5
from keras.models import load_model
model=load_model('save_models/dr_cnn.h5')
test_eval=model.evaluate(test_array,y_test,verbose=1)
print('Test loss:',test_eval[0])
print('Test accuracy:',test_eval[1])
11/11 [==============================] - 0s 14ms/step - loss: 1.0253 -
accuracy: 0.5937
Test loss: 1.025274395942688
Test accuracy: 0.5936599373817444

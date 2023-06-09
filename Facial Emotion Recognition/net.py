







import cv2
import math
import numpy as np
import pandas as pd

import scikitplot
import seaborn as sns
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from PIL import  Image
import matplotlib.pyplot as plt
from keras.utils import np_utils








import cv2
from tensorflow import keras
import numpy as np
from PIL import  Image
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import optimizers
from keras.models import model_from_json
from keras.metrics import categorical_accuracy
from keras.models import load_model



from tensorflow.python.keras.models import model_from_yaml




yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)

# load weights into new model
model.load_weights('model.h5')
model.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy'])

filname = 'fer2013.csv'
label_map = ('Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

df = pd.read_csv(filname)
print(df.shape)
df.head()

df.emotion.unique()

df.emotion.value_counts()


math.sqrt(len(df.pixels[0].split(' ')))


INTERESTED_LABELS = [0, 1, 2, 3, 4,5, 6]

df = df[df.emotion.isin(INTERESTED_LABELS)]
df.shape

img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0)


img_array.shape



le = LabelEncoder()
img_labels = le.fit_transform(df.emotion)
img_labels = np_utils.to_categorical(img_labels)
img_labels.shape


le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)



X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                    shuffle=True, stratify=img_labels,
                                                    test_size=0.1, random_state=42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape



del df
del img_array
del img_labels



img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1]


X_train = X_train / 255.
X_valid = X_valid / 255.












score = model.predict_classes(X_valid)
print (model.summary())

#new_X = [ np.argmax(item) for item in score ]
y_test2 = [ np.argmax(item)for item in y_valid]

# Calculating categorical accuracy taking label having highest probability
accuracy = [ (x==y) for x,y in zip(score,y_test2) ]
print(" Accuracy on Test set : " , np.mean(accuracy))





from tkinter import *
from PIL import Image , ImageTk
from tkinter import filedialog, Label
import os
from tkinter import messagebox
photo=""
result_=""
def showimage():

    for widget in frame.winfo_children():
        widget.destroy()

    fln=filedialog.askopenfilename(initialdir=os.getcwd(),title="Select Image File",filetypes=(("JPG file ","*.jpg" ),("PNG file ","*.png" ),("All File ","*.*")))
    photo=fln
   # print(fln)

    img=Image.open(fln)
    img.thumbnail((500,500))
    img=ImageTk.PhotoImage(img)


    lb1 = Label(frame,text=img,bg="gray")
    lb1.pack()
    lb1.configure(image=img)
    lb1.image=img
    imgGray = cv2.imread(fln, 0)

    try:
        imgGray = cv2.resize(imgGray, (48, 48), interpolation=cv2.INTER_AREA)


    except:
        print("An exception occurred")

    # imgGray = Image.open('4.jpeg').convert("L")
    # img=imgGray

    # imgGray = cv2.resize(imgGray, (48, 48))

    arr = np.asarray(imgGray)

    data = np.asarray(imgGray)

    data = np.array(data)
    h, w = data.shape
    one_dim = data.flatten()

    image = one_dim;
    re = [];
    # re.append([int(p) for p in image.split()]);
    # arr=np.array(re)
    # arr = arr.reshape( 48, 48)
    re = np.array(image) / 255.0;
    re = re.reshape(1, h, w, 1);



    new_predect_value = model.predict_classes(re)

    # new_predect_value=np.argmax(predect_value);
    # new_predect_value=np.mean(new_predect_value);

    if (new_predect_value == 0.0):
        result_ = 'Anger'
    if (new_predect_value == 1.0):
        result_ = 'Disgust'
    if (new_predect_value == 2.0):
        result_ = 'Fear'
    if (new_predect_value == 3.0):
        result_ = 'Happy'
    if (new_predect_value == 4.0):
        result_ = 'Sad'
    if (new_predect_value == 5.0):
        result_ = 'Surprise'
    if (new_predect_value == 6.0):
        result_ = 'Neutral'
    # print(" result : ", result_);
    messagebox.showinfo("emotion", result_)
"""
def finel_res():
    print(result_)
    messagebox.showinfo("emotion", result_)
"""

top=Tk()
top.title("Load Image")
can=Canvas(top,height=600,width=600,bg="#263D42")
can.pack()


frame=Frame(top,bg="white")
frame.place(relwidth=0.8,relheight=0.8,relx=0.1,rely=0.1)



btn=Button(top,text='Browse',padx=10,pady=5,fg="white",bg="#263D42",command=showimage)
btn.pack()



#res_image=Button(top,text='Result',padx=10,pady=5,fg="white",bg="#263D42",command=finel_res)
#res_image.pack()



top.mainloop()






#plt.imshow(img, cmap='gray', vmin=0, vmax=255)
#plt.show()


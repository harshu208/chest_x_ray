import numpy as np 
import pandas as pd 
from glob import glob
%matplotlib inline
import matplotlib.pyplot as plt
import os
import sklearn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Input
from keras.layers import Conv2D, Activation, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input, ResNet50
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/data/Data_Entry_2017.csv')
image_path = {os.path.basename(x): x for x in 
                   glob(os.path.join('..', 'input/data/', 'images*', '*', '*.png'))}
#print(image_path)
#print(data.columns)
data['path'] = data['Image Index'].map(image_path.get)
print(data.head(5))

print(len(data))
# Total number of image_path
print(len(image_path))

disease_counts = data.groupby('Finding Labels')['Image Index'].count().sort_values(ascending=False).iloc[:10]
print(disease_counts)

# plotting top 10 labels' count
plt.figure(figsize=(12,8))
plt.bar(np.arange(len(disease_counts))+0.5, disease_counts, tick_label=disease_counts.index)
plt.xticks(rotation=90)

data['Finding Labels'] = data['Finding Labels'].map(lambda x: x.replace('No Finding','Nothing'))
#print(data['Finding Labels'])
from itertools import chain
labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
print(labels)

print(data['Finding Labels'])

for lbl in labels: 
    data[lbl] = data['Finding Labels'].map(lambda find: 1 if lbl in find else 0)
    
#print(data['Cardiomegaly'])
data['encoding'] = [[1 if l in lbl.split('|') else 0 for l in labels] for lbl in data['Finding Labels']]

print(data['encoding'])
#print(data[['encoding','Finding Labels']])

print(data.head(5))

class_count = {}
for lbl in labels:
    class_count[lbl] = data[lbl].sum()
#print(class_count)
classweight = {}
for lbl in labels :
    classweight[lbl] = 1/class_count[lbl]
#print(classweight['Nothing'])
classweight['Nothing'] /= 2   #Extra penalising the none class 
#print(classweight['Nothing'])
#print(classweight)
def func(row):
    weight = 0
    for lbl in labels: 
        if(row[lbl]==1):
            weight += classweight[lbl]
    return weight
new_weights = data.apply(func, axis=1)
#print(new_weights)
sampled_data = data.sample(40000, weights = new_weights)
#sampled_data = data.sample(40000)
print(sampled_data['Nothing'].sum())

sampled_data.to_csv('sampled_data.csv')

disease_counts = sampled_data.groupby('Finding Labels')['Image Index'].count().sort_values(ascending=False).iloc[:20]

# plotting top 10 labels' count
plt.figure(figsize=(12,8))
plt.bar(np.arange(len(disease_counts))+0.5, disease_counts, tick_label=disease_counts.index)
plt.xticks(rotation=90)

from sklearn.model_selection import train_test_split
train_data , test_data = train_test_split(sampled_data, test_size=0.2)
train_data , valid_data = train_test_split(train_data, test_size=0.25)
print(len(train_data))
print(len(test_data))
print(len(valid_data))

from keras.preprocessing.image import ImageDataGenerator 
IMG_SIZE = (299, 299)
# core imagedatagenerator used to create train and test imageDatagenerators
core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)


valid_data['newLabel'] = valid_data.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
#print(valid_data['newLabel'])
train_data['newLabel'] = train_data.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
test_data['newLabel'] = test_data.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
train_gen = core_idg.flow_from_dataframe(
    dataframe=train_data,
    directory=None,
    x_col = 'path',
    y_col = 'newLabel',
    class_mode = 'categorical',
    target_size = IMG_SIZE,
    color_mode = 'rgb',
    batch_size = 32)
print(type(train_gen))
#print(train_gen.length())
valid_gen = core_idg.flow_from_dataframe(
    dataframe=valid_data,
    directory=None,
    x_col = 'path',
    y_col = 'newLabel',
    class_mode = 'categorical',
    target_size = IMG_SIZE,
    color_mode = 'rgb',
    batch_size = 256) # we can use much larger batches for evaluation

    valid_x,valid_y = next(core_idg.flow_from_dataframe(
    dataframe=valid_data,
    directory=None,
    x_col = 'path',
    y_col = 'newLabel',
    class_mode = 'categorical',
    target_size = IMG_SIZE,
    color_mode = 'rgb',
    batch_size = 256))
test_X, test_Y = next(core_idg.flow_from_dataframe(
    dataframe=test_data,
    directory=None,
    x_col = 'path',
    y_col = 'newLabel',
    class_mode = 'categorical',
    target_size = IMG_SIZE,
    color_mode = 'rgb',
    batch_size = 1024))
#print(len(test_X))
#print(len(test_X[0]))

#print(test_X.shape)


t_x,t_y=next(train_gen)
print(t_x.shape)
print(t_y.shape)
'''
i=0
while(1):
    t_x,t_y = next(train_gen)
    i = i+1
    if(i>500):
        print(32*i)'''
    

fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
#print(fig)
#print(m_axs.flatten())
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -1.5, vmax = 1.5)
    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(labels, c_y) 
                             if n_score==1]))
    c_ax.axis('off')


from keras.applications.resnet50 import ResNet50, preprocess_input

base_model = ResNet50(weights='imagenet',include_top=False)

print(base_model.summary())

n_classes = len(labels)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='sigmoid')(avg)
model = keras.Model(inputs=base_model.inputs, outputs = output)

print(model.summary())

met = ['categorical_accuracy', keras.metrics.Precision(), keras.metrics.AUC(), 'binary_accuracy']


pip install focal-loss

from focal_loss import BinaryFocalLoss

for layer in base_model.layers:
    layer.trainable = True
model.compile(loss=BinaryFocalLoss(gamma=2), optimizer='nadam', metrics=met)

history = model.fit_generator(train_gen, steps_per_epoch=10, validation_data=(valid_x, valid_y), epochs=3,max_queue_size=100, workers=-1, use_multiprocessing=True)

def plot_roc():
    pred_Y =  model.predict(test_X, batch_size = 32)
    from sklearn.metrics import roc_curve, auc
    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
    sum=0;
    for (idx, c_label) in enumerate(labels):
        print(pred_Y[:,idx])
        fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
        sum+=auc(fpr,tpr)
    print(sum/15)
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    fig.savefig('XceptionRoc.png')

plot_roc();
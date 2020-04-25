#-*- coding: utf-8 -*-
# 使用するGPUを指定するには環境変数 CUDA_VISIBLE_DEVICES に0-3の値を指定して実行すること。
import keras
import tensorflow
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback
from keras.optimizers import SGD, rmsprop
from keras.regularizers import l2
from keras.utils import to_categorical
import keras.backend as K
import sys, glob
import os
import zipfile, io, re
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

# パラメータ

#steps_per_epoch = int(X.shape[0] // batch_size)
#validation_steps = int(count_validation / batch_size)
# Trueなら学習せず、predictのみ実行する
predict_only = True 
#use_ensemble = False
# True なら generator を使う。Falseならgeneratorを使わずdatasetをそのまま渡す
use_generator = True
# cross validation をつかう(True)か、1回だけ(False)か
use_cross_validation = True
# お試しでMNISTを対象に実行するか
use_mnist = False
#use_cross_validation = False
use_simple = False
#use_simple = True
use_down_sampling=True

batch_size=4
epochs=10
# use_cross_validation = True のときのみ使う。分割回数指定
n_splits=4
# 各splitで、何回ダウンサンプルして学習をやり直すか？
#(クロスバリデーションの1splitあたりの学習エポック数は sub_split * epoch になる。
sub_split=10

if use_simple:
  batch_size=4
  epochs=2
  n_splits=2
  sub_split=2

def down_sample(x,y, shuffle=True):
   global use_down_sampling
   if not use_down_sampling:
     # ダウンサンプリングしない版
     r = np.arange(len(y))
     if shuffle: r=np.random.permutation(r)
     return r
   categories = np.unique(y)
   indices = {}
   for c in categories:
     indices[c] = np.where(y == c)[0]
   each_num = min([len(indices[i]) for i in indices])
   res = []
   for c in categories:
     index = np.random.choice(indices[c], each_num, replace=False) # 重複なしeach_num個のランダム抽出
     res.extend(index)
   # シャッフル
   if shuffle:
      res = np.random.permutation(res)
   print("size change ", len(y), ' -> ', len(res), '(min',each_num,')')
   return res
K.clear_session()

if use_mnist:
  img_dir = '/path/to/MNIST/Data/Train'
  img_dir_test = '/path/to/MNIST/Data/Validation'
  classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  file_ext = 'jpg'
  image_size=32
else:
  img_dir = '/path/to/dataset'
  img_dir_test = '/path/to/test-dataset'
  classes = ["Papillotubularcarcinoma", "Scirrhouscarcinoma", "Solid-tubularcarcinoma"]
  file_ext = 'png'
  image_size=300

img_height=image_size
img_width=image_size

num_classes = len(classes)

def load_dataset(path, classes):
  global image_size
  X=[]
  Y=[]
  Files=[]
  for index, ClassLabel in enumerate(classes):
    global use_generator
    ImagesDir = os.path.join(path, ClassLabel)
    # RGB変換
    print(index)
    print(ClassLabel)
    print(ImagesDir)
    files = glob.glob(ImagesDir+"/*."+file_ext)
    for i, file in enumerate(files):
        print(i)
        print(file)
        image = Image.open(file)
        image = image.convert('RGB')
        # リサイズ
        image = image.resize((image_size, image_size))
        # 画像から配列に変換 
        data = np.asarray(image)
        X.append(data)
        Y.append(index)
        Files.append(file)
  X = np.array(X)
  Y = np.array(Y)
  # データ型の変換＆正規化
  if not use_generator:
    X = X.astype('float32') / 255
  return (X, Y, np.array(Files))
X, Y, _ = load_dataset(img_dir, classes)
Xtest, Ytest, testFiles = load_dataset(img_dir_test, classes)

print(X.shape, Y.shape)

dir_tf_log = '../tf_log'

# セッションの初期化
config = tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(allow_growth=True))
session = tensorflow.Session(config=config)
K.tensorflow_backend.set_session(session)

#訓練データ拡張
horizontal_flip = True
vertical_flip = True
if use_mnist:
  # mnist は反転すると数字が変わるので反転オフにする
  horizontal_flip = False
  vertical_flip = False
datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.1,
        brightness_range=[0.9,1.0],
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        zoom_range=False,
        #channel_shift_range=30,
        fill_mode='reflect')

def create_model():
  global image_size
  #VGGモデルの読み込み。最終層は読み込まない
  input_shape = (image_size, image_size, 3)
  base_model = VGG16(weights= 'imagenet', include_top=False, input_shape=input_shape)
  #最終層の設定
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  #x = Dense(1024, activation='relu')(x)
  predictions = Dense(num_classes, activation = 'softmax')(x)
  m = Sequential()
  m.add(Flatten(input_shape=base_model.output_shape[1:]))
  m.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
  m.add(BatchNormalization())
  m.add(Dropout(0.5))
  m.add(Dense(num_classes, activation='softmax'))
  predictions = m(base_model.output)

  model = Model(inputs=base_model.input, outputs=predictions)
  # #FineTuning
  # for layer in model.layers[:14]:
  #   layer.trainable = False
  # for layer in model.layers[14:]:
  #   layer.trainable = True
  # model.summary()

  # 学習率
  #opt = SGD()
  opt = SGD(lr=0.001)
  #opt = rmsprop(lr=5e-7, decay=5e-5)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  #model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model

# 出力ファイル名
model_save_name = '../h5files/SGD_ensemble_%d.hdf5'
csvlog_name = '../log/SGD.log'

# コールバック
#checkpointer = ModelCheckpoint(filepath=model_save_name, verbose=1, save_best_only=True)
csv_logger = CSVLogger(csvlog_name)

# 学習
script_path = os.path.abspath(__file__)

skf = KFold(n_splits=n_splits, shuffle=True)
index = 0
test_pred = []
if not predict_only:
	train_val = []
	if use_cross_validation:
		train_val = skf.split(X, Y)
	else:
		i = np.random.permutation(np.arange(len(X)))
		s = int(len(i) * 3 / 4) + 1
		train_val = [(i[0:s], i[s:])]
	for train, val in train_val:
		model = create_model()
		if index == 0: model.summary()
		model_name = model_save_name % index
		index += 1
		checkpointer = ModelCheckpoint(filepath=model_name, monitor='val_loss', verbose=1, save_best_only=True)
		for i2 in np.arange(sub_split):
			print("begin split ",index,'/',n_splits,", subsplit ",i2+1,'/',sub_split)
			i = down_sample(X[train], Y[train])
			x, y = X[train][i], Y[train][i]

			y = [to_categorical(i, num_classes=num_classes) for i in y]
			
			i = down_sample(X[val], Y[val])
			xval, yval = X[val][i], Y[val][i]
			yval = [to_categorical(i, num_classes=num_classes) for i in yval]
			
			i = down_sample(Xtest, Ytest)
			xtest, ytest = Xtest[i], Ytest[i]
			ytest = [to_categorical(i, num_classes=num_classes) for i in ytest]
			x = np.array(x);       y = np.array(y)
			xval = np.array(xval); yval = np.array(yval)
			xtest = np.array(xtest); ytest = np.array(ytest)
			if use_generator:
				history = model.fit_generator(
					    datagen.flow(x, y, batch_size = batch_size),
	        		            steps_per_epoch=x.shape[0] // batch_size,
	        		            epochs=epochs,
	        		            validation_data = datagen.flow(xval, yval, batch_size = batch_size),
	        		            validation_steps=xval.shape[0] // batch_size,
	        		            verbose=1,
	        		            callbacks=[csv_logger, checkpointer])
			else:
				history = model.fit(x=x, y=y, batch_size=batch_size,
					epochs=epochs, verbose=1,
					validation_data = (xval, yval),
					callbacks=[csv_logger, checkpointer])
	# Evaluation
		if use_generator:
			scores = model.evaluate_generator(datagen.flow(xtest, ytest, batch_size=batch_size), verbose=0)
		else:
			scores = model.evaluate(x=xtest, y=ytest, batch_size=batch_size, verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		test_pred.append(scores[1])
		K.clear_session()

preds=[]
models=[]
# 学習済みモデルを使って予測を行う。
# x: 予測対象のリスト
# raw: Trueなら予測結果リストを集計せずに返す(modelごとの予測結果をcategoricalで返す)。デフォルトFalse
# model_range: 利用するモデルを選択する。たとえばmodel_range=[0,2] なら1番目と3番目を使う。Noneならすべて使う(np.arange(n_split)と同じ)。デフォルトはNone。
def predict(x, raw=False, model_range=None, use_argmax=True):
    global model_save_name, n_splits, preds, use_cross_validation, predict_only
    pred = []
    default_model_range = np.arange(n_splits if use_cross_validation else 1)
    if len(models) == 0: # キャッシュがなければキャッシュを作る
      for i in default_model_range:
          model = create_model()
          model.load_weights(model_save_name % i)
          models.append(model)
      if predict_only:
          models[0].summary()

    model_range = np.array(model_range) if model_range else default_model_range
    for i in model_range:
        model = models[i]
        p = model.predict(x)
        pred.append(p)
    if not raw:
      pred = np.array(pred)
      pred = np.sum(pred, axis=0)
      if use_argmax:
        pred = np.argmax(pred, axis=1)
    preds.append(pred)
    return np.array(pred)
print("Predict:")




for i in np.arange(num_classes):
    x = Xtest[Ytest == i]
    files = testFiles[Ytest == i]
    # generator で正規化している場合、データ自体が正規化されてないので、先に正規化する
    if use_generator:
       x = x / 255.
    
    result = predict(x, use_argmax=False)
    #result = predict(x, use_argmax=False, model_range=[1])
    for k,file in enumerate(files):
       print("class ", i,"\tpredict ", np.argmax(result[k]), "\t", file, "\t", result[k]) 
    #print("class ", i, ": ", predict(x, use_argmax=False))
#print("average acc: %.2f%%" % (np.mean(np.array(test_pred))*100))
#K.clear_session()


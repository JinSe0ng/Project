import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0

# 데이터 로딩 및 전처리
gen = ImageDataGenerator(rescale=1./255, validation_split=0.2, zoom_range=0.99)

train_data = gen.flow_from_directory("/Users/ijinseong/Documents/for_git/LangChain/Project_4/Brain Tumor Data Set",
                                    target_size=(150, 150),
                                    batch_size=32,
                                    class_mode="binary",
                                    subset="training",
                                    shuffle=True,
                                    seed=123)

val_data = gen.flow_from_directory("/Users/ijinseong/Documents/for_git/LangChain/Project_4/Brain Tumor Data Set",
                                  target_size=(150, 150),
                                  batch_size=32,
                                  class_mode="binary",
                                  subset="validation",
                                  shuffle=True,
                                  seed=123)

# 모델 설계
model = models.Sequential([
    EfficientNetB0(include_top=False, input_shape=(150, 150, 3), weights='imagenet'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 콜백 설정 (EarlyStopping, 모델 체크포인트)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# 모델 학습
history = model.fit(train_data, validation_data=val_data, epochs=20, callbacks=[early_stopping, checkpoint])

# 모델 평가 및 시각화
test_loss, test_accuracy = model.evaluate(val_data)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# 훈련/검증 정확도 그래프
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# 혼동 행렬 및 분류 보고서
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred = model.predict(val_data)
y_pred_classes = (y_pred > 0.5).astype('int32')
y_true = val_data.classes

conf_matrix = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print(classification_report(y_true, y_pred_classes))

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from google.colab import drive

# 1. Kết nối với Google Drive để lấy dữ liệu
drive.mount('/content/drive')

# 2. Giải nén file ảnh (Đảm bảo file Processed_Data.zip nằm ngoài cùng trên Drive)
!unzip -q "/content/drive/MyDrive/Processed_Data.zip" -d "/content/dataset"

# 3. Nạp dữ liệu và chia tập Huấn luyện (80%) - Kiểm thử (20%)
DATA_DIR = "/content/dataset/Processed_Data"
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='validation'
)

# 4. Xây dựng mô hình AI bằng MobileNetV2 (Transfer Learning)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Đóng băng các lớp gốc để giữ kiến thức định dạng ảnh

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
# Lớp xuất kết quả bằng đúng số lượng loại bệnh lá lúa
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Tiến hành huấn luyện (Chạy 10 vòng - Epochs)
print("Bắt đầu huấn luyện AI...")
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# 6. Lưu mô hình (Bộ não AI) về Google Drive
model.save("/content/drive/MyDrive/model_lalua.h5")
print("✅ Đã lưu file mô hình 'model_lalua.h5' vào Google Drive!")

# 7. Vẽ biểu đồ đánh giá độ chính xác
plt.plot(history.history['accuracy'], label='Độ chính xác (Train)')
plt.plot(history.history['val_accuracy'], label='Độ chính xác (Validation)')
plt.title('Biểu đồ đánh giá Mô hình')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

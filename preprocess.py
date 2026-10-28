import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# 1. Cấu hình đường dẫn
INPUT_FOLDER = "Raw_Data"         # Thư mục chứa dữ liệu gốc tải từ Kaggle
OUTPUT_FOLDER = "Processed_Data"  # Thư mục sẽ chứa dữ liệu sau khi xử lý
IMG_SIZE = 224                    # Kích thước chuẩn cho mô hình MobileNetV2
AUGMENT_MULTIPLIER = 3            # Số lượng ảnh sinh thêm từ 1 ảnh gốc (để chống học vẹt)

# Tạo thư mục đầu ra nếu chưa có
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# 2. Cấu hình công cụ Data Augmentation (Tăng cường dữ liệu)
datagen = ImageDataGenerator(
    rotation_range=30,      # Xoay ảnh ngẫu nhiên tối đa 30 độ
    width_shift_range=0.1,  # Dịch chuyển ngang
    height_shift_range=0.1, # Dịch chuyển dọc
    horizontal_flip=True,   # Lật ngang ảnh
    brightness_range=[0.8, 1.2] # Chỉnh độ sáng ngẫu nhiên từ tối đến sáng
)

print("🚀 Bắt đầu quá trình tiền xử lý và tăng cường dữ liệu...")

# 3. Quét qua từng thư mục bệnh (class) trong Raw_Data
for class_name in os.listdir(INPUT_FOLDER):
    class_path = os.path.join(INPUT_FOLDER, class_name)
    
    # Bỏ qua nếu không phải là thư mục
    if not os.path.isdir(class_path): continue

    # Tạo thư mục tương ứng trong Processed_Data
    out_class_path = os.path.join(OUTPUT_FOLDER, class_name)
    if not os.path.exists(out_class_path):
        os.makedirs(out_class_path)
        
    print(f"Đang xử lý thư mục: {class_name}...")
    
    # 4. Xử lý từng ảnh trong thư mục bệnh
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        try:
            # Đọc và Resize ảnh về 224x224
            img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1,) + img_array.shape) # Reshape để phù hợp với hàm datagen
            
            # Lưu ảnh gốc đã resize vào Processed_Data
            img_resized = cv2.imread(img_path)
            img_resized = cv2.resize(img_resized, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(out_class_path, f"resized_{img_name}"), img_resized)
            
            # Sinh thêm ảnh mới (Augmentation)
            i = 0
            for batch in datagen.flow(img_array, batch_size=1, 
                                      save_to_dir=out_class_path, 
                                      save_prefix='aug', 
                                      save_format='jpg'):
                i += 1
                if i >= AUGMENT_MULTIPLIER:
                    break # Dừng lại khi đã sinh đủ số ảnh yêu cầu
        except Exception as e:
            print(f"Lỗi ảnh {img_name}: {e}")

print("✅ Đã hoàn thành! Toàn bộ dữ liệu sạch nằm trong thư mục 'Processed_Data'")
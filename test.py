import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from PIL import Image, ImageTk

# Xác định tên lớp
class_names = ['00000', '10000', '20000', '50000']

# Tải mô hình VGG16 đã được huấn luyện trước
def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

   # Đóng băng các lớp
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Create model
    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)


    # Thêm lớp FC và Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(len(class_names), activation='softmax', name='predictions')(x)

    # Compile model
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

# Load trained model weights
my_model = get_model()
my_model.load_weights("weights-13-0.92.hdf5")

# Chức năng dự đoán lớp của ảnh
def predict_image(image):
    try:
        image = cv2.resize(image, (128, 128))
        image = image.astype('float') / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = my_model.predict(image)
        max_index = np.argmax(prediction)
        if prediction[0][max_index] >= 0.8:
            class_label = class_names[max_index]
            confidence = np.max(prediction[0])
            return class_label, confidence
        else:
            return "Unknown Class", 0.0
    except Exception as e:
        print("Error occurred during image prediction:", str(e))
        return None, 0.0

# Chức năng chọn ảnh từ máy tính
def choose_image():
    try:
        image_path = filedialog.askopenfilename(title="Choose Image", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if image_path:
            image = cv2.imread(image_path)
            if image is not None:
                class_label, confidence = predict_image(image)
                if class_label is not None:
                    result_label.config(text="Giá trị dự đoán: {}\nConfidence: {:.2f}".format(class_label, confidence))
                    display_image(image)
                else:
                    print("Failed to predict image.")
            else:
                print("Failed to read image.")
    except Exception as e:
        print("Error occurred during image selection:", str(e))

#Chức năng hiển thị hình ảnh trên giao diện
def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((300, 300), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

# Create GUI
window = tk.Tk()
window.title("Dự đoán tiền Việt Nam")
window.geometry("400x400")

# Tạo nút chọn ảnh
choose_button = tk.Button(window, text="Choose Image", command=choose_image)
choose_button.pack(pady=10)

#Tạo nhãn ảnh
image_label = tk.Label(window)
image_label.pack()

# Tạo nhãn kết quả
result_label = tk.Label(window)
result_label.pack()

window.mainloop()
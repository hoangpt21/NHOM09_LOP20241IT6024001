import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow import keras
from keras.models import load_model 
from PIL import Image, ImageChops, ImageEnhance

# Function to convert to ELA image
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

# Function to prepare image for prediction
image_size = (128, 128)
def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

# Function to handle button click for loading image
def load_image():
    file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize the image to fit in the interface
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        global loaded_image_path
        loaded_image_path = file_path
    result_label.config(text='')

# Function to handle button click for running the model
class_names = ['fake', 'real']
def run_model():
    if loaded_image_path:
        model = load_model('./model_casia_run1.h5')
        image = prepare_image(loaded_image_path)
        image = image.reshape(-1, 128, 128, 3)
        y_pred = model.predict(image)
        y_pred_class = np.argmax(y_pred, axis = 1)[0]
        result_label.config(text=f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')
    else:
        result_label.config(text='No image loaded. Please load an image first.')

# Create the main window
root = tk.Tk()
root.title("Fake and real image classification")
root.minsize(600, 400)
# Create and configure widgets
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

run_button = tk.Button(root, text="Dự đoán", command=run_model)
run_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Global variable to store the loaded image path
loaded_image_path = None

# Run the Tkinter event loop
root.mainloop()

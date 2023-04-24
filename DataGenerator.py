import numpy as np
import os
from PIL import Image

# Load and preprocess the images
def load_and_preprocess_images(image_dir, img_size=(512, 128)):
    images = []
    labels = []
    
    for label in os.listdir(image_dir):
        label_dir = os.path.join(image_dir, label)
        #print(label_dir)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                if img_file.endswith(('.png', '.jpeg')):
                    img_path = os.path.join(label_dir, img_file)
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = img.resize(img_size, Image.Resampling.LANCZOS)  # Resize the image
                    img_array = np.array(img) / 255.0  # Normalize the image
                    images.append(img_array)
                    labels.append(int(label))
    
    images = np.array(images)
    labels = np.array(labels)

    np.save('images.npy', images)
    np.save('labels.npy', labels)
    return images, labels


src_dir = "C:/Users/eenysh/Desktop/Data_collection(20221011)/dataset_folder"
X, y = load_and_preprocess_images(src_dir)
X = X.reshape(X.shape[0], 512, 128, 1)  # Add the channel dimension for grayscale images

X_f = np.repeat(X, 3, axis=-1)
X_f = np.float32(X_f)
print(np.shape(X_f))
print(X_f.dtype)
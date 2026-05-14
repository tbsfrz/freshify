import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

img_path = "2.jpg"
model = MobileNetV2(weights="imagenet")

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
results = decode_predictions(preds, top=5)[0]

for imagenet_id, label, score in results:
    print(f"{label}: {score:.4f}")
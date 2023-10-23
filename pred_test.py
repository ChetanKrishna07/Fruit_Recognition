import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def pred_and_plot(model, folder, classes):
  imgs = os.listdir(folder)
  plt.figure(figsize=(15, 5))
  sub_s = int(len(imgs) / 2)
  for i in range(0, len(imgs)):
    plt.subplot(2, sub_s, i + 1)
    img = tf.io.read_file(os.path.join(folder, imgs[i]))
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.
    preds = model.predict(tf.expand_dims(img, axis=0), verbose=0)
    pred = np.argmax(preds)
    class_name = classes[pred]
    conf=preds[0][pred]*100
    plt.imshow(img)
    plt.title(f"{class_name} ({conf:2.2f}%)")
    plt.axis("off")
  plt.show()

model = tf.keras.models.load_model("fruit_quality_model_resnet")
print("model imported")

classes = np.array(['freshapples', 'freshbanana', 'freshoranges', 'rottenapples',
       'rottenbanana', 'rottenoranges'])

print(classes)
pred_and_plot(model, "./test_images/", classes=classes)

print("plotted")


import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape=(224,224,3), base_trainable=False):
    base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape, pooling="avg")
    base.trainable = base_trainable
    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="cnn_image_only")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC","Precision","Recall"])
    return model

def build_hybrid_cnn(img_shape=(224,224,3), meta_dim=0, base_trainable=False):
    img_in = layers.Input(shape=img_shape, name="image")
    base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=img_shape, pooling="avg")
    base.trainable = base_trainable
    x = tf.keras.applications.efficientnet.preprocess_input(img_in)
    x = base(x)
    x = layers.Dropout(0.2)(x)
    if meta_dim and meta_dim > 0:
        meta_in = layers.Input(shape=(meta_dim,), name="meta")
        m = layers.Dense(64, activation="relu")(meta_in)
        m = layers.Dropout(0.2)(m)
        fused = layers.Concatenate()([x, m])
        out = layers.Dense(1, activation="sigmoid")(fused)
        model = models.Model([img_in, meta_in], out, name="cnn_hybrid")
    else:
        out = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(img_in, out, name="cnn_image_only")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC","Precision","Recall"])
    return model

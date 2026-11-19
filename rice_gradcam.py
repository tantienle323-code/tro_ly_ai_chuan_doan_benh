import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    class_index: int,
    last_conv_layer_name: str = "Conv_1",
) -> np.ndarray:
    feature_extractor = next(
        (
            layer
            for layer in model.layers
            if layer.name == "rice_feature_extractor"
        ),
        None,
    )
    if feature_extractor is None:
        grad_model = tf.keras.Model(
            model.input,
            [model.get_layer(last_conv_layer_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            class_score = predictions[:, class_index]
    else:
        feature_index = model.layers.index(feature_extractor)
        preprocessing_model = tf.keras.Model(
            model.input,
            model.layers[feature_index - 1].output,
        )
        with tf.GradientTape() as tape:
            features = preprocessing_model(img_array, training=False)
            conv_output, features = feature_extractor(features, training=False)
            for layer in model.layers[feature_index + 1 :]:
                features = layer(features, training=False)
            class_score = features[:, class_index]

    gradients = tape.gradient(class_score, conv_output)
    if gradients is None:
        raise ValueError("Không tính được gradient cho lớp tích chập.")
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = conv_output[0] @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.maximum(tf.squeeze(heatmap), 0)
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    return heatmap.numpy()

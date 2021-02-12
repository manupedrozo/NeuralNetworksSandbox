import tensorflow as tf

# Intersection over union for each class in the batch.
# Then compute the final iou as the mean over classes
def gen_meanIoU(num_classes):

    def meanIoU(y_true, y_pred):
        # get predicted class from softmax
        y_pred = tf.expand_dims(tf.argmax(y_pred, -1), -1)

        per_class_iou = []

        for i in range(1, num_classes): # exclude the background class 0
            # Get prediction and target related to only a single class (i)
            class_pred = tf.cast(tf.where(y_pred == i, 1, 0), tf.float32)
            class_true = tf.cast(tf.where(y_true == i, 1, 0), tf.float32)
            intersection = tf.reduce_sum(class_true * class_pred)
            union = tf.reduce_sum(class_true) + tf.reduce_sum(class_pred) - intersection

            iou = (intersection + 1e-7) / (union + 1e-7)
            per_class_iou.append(iou)

        return tf.reduce_mean(per_class_iou)

    return meanIoU
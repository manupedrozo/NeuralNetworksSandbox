import os
import tensorflow as tf

"""
Common tensorflow callback setups
"""

def tensorboard(experiment_dir):
    """Enable TensorBoard logging"""
    tb_dir = os.path.join(experiment_dir, 'tb_logs')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
        
    return tf.keras.callbacks.TensorBoard(
        log_dir=tb_dir,
        histogram_freq=1,
    )

def early_stopping(patience):
    """Enable early stopping on validation loss with the given patience, restore the best weights after stopping."""
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=patience,
        restore_best_weights=True,
    )

def checkpoints(experiment_dir, now_string):
    """Enable checkpoints for each epoch"""
    ckpt_dir = os.path.join(experiment_dir, 'ckpts')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}_' + f'{now_string}.ckpt'), 
        save_weights_only=True,
    )

def save_best(experiment_dir):
    """Enable best model checkpoint which is overwritten when a new best model is found (min validation loss)."""
    best_dir = os.path.join(experiment_dir, 'best')
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    checkpoint_path = os.path.join(best_dir, f'model')

    return checkpoint_path, tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
    )

from model import get_model
from data import load_and_prepare_for_mlm

import tensorflow as tf

from os.path import join

if __name__=="__main__":    
    
    model=get_model()
    tf_dataset = load_and_prepare_for_mlm(join('data', 'book'))
    tf_val_dataset = load_and_prepare_for_mlm(join('data', 'book_val'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss)#model.compute_loss)  # Use the model's compute_loss method
    model.fit(tf_dataset, epochs=3,validation_data=tf_val_dataset,)  # Adjust the number of epochs as needed
    
    model.save('bert_model')

from datasets import  load_dataset  , load_from_disk
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer
from os.path import join



tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
mask_token_id = tf.constant(tokenizer.mask_token_id, dtype=tf.int64)#tokenizer.mask_token_id
vocab_size = len(tokenizer)

def tokenize_and_encode(examples):
    # Tokenize and encode sequences in the dataset
    encoded = tokenizer(examples['text'], return_tensors='tf', truncation=True, padding='max_length', max_length=128, return_special_tokens_mask=True)

    # TensorFlow expects a 'dict' of inputs
    inputs = {'input_ids': encoded['input_ids'], 'token_type_ids': encoded['token_type_ids'], 'attention_mask': encoded['attention_mask']}
    return inputs

def load_and_prepare_for_mlm(path):
    def dynamic_masking(features):
        # Function to dynamically apply masking
        input_ids, attention_mask = features['input_ids'], features['attention_mask']
        # Create a copy of input_ids to work with
        labels = tf.identity(input_ids)

        # Determine the probability of masking
        mask_probability = 0.15

        # Create a random array of floats with the same shape as input_ids
        rand = tf.random.uniform(shape=tf.shape(input_ids))

        # Determine where to apply the masks
        # Do not mask special tokens (CLS, SEP, PAD)
        mask_cond = (rand < mask_probability) & (input_ids != tokenizer.cls_token_id) & \
                    (input_ids != tokenizer.sep_token_id) & (input_ids != tokenizer.pad_token_id)
        mask_indices = tf.where(mask_cond)

        # Apply mask: Set input_ids to mask_token_id where mask_cond is True
        updates = tf.fill(tf.shape(mask_indices)[0:1], mask_token_id)
        masked_input_ids = tf.tensor_scatter_nd_update(input_ids, mask_indices, updates)

        # Set labels for non-masked tokens to -100 so they are not used in the loss computation
        labels_mask = 1 - tf.cast(mask_cond, tf.int64)
        labels = labels * labels_mask + -100 * (1 - labels_mask)

        return {"input_ids": masked_input_ids, "attention_mask": attention_mask, "token_type_ids": features['token_type_ids']}, labels

    # Load the tokenized dataset from disk
    dataset = load_from_disk(path)
    dataset.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'token_type_ids'])

    # Convert to TensorFlow dataset
    features = {x: np.array(dataset[x]) for x in ['input_ids', 'attention_mask', 'token_type_ids']}
    tf_dataset = tf.data.Dataset.from_tensor_slices(features)

    # Apply dynamic masking
    tf_dataset = tf_dataset.map(dynamic_masking)

    return tf_dataset.batch(32)  # Batch the dataset

val_size = 1_000_000

if __name__=="__main__":
    dataset = load_dataset("bookcorpus", split=f"train[{val_size}:]")
    # Apply the function to the whole dataset
    dataset = dataset.map(tokenize_and_encode, batched=True)
    #dataset = tf_dataset.with_format('tensorflow')
    dataset.save_to_disk(join('data','book'))
    

    #NO VAL DATA making my own
    val_dataset = load_dataset("bookcorpus", split="train[:{val_size}]")
    # Apply the function to the whole dataset
    val_dataset = val_dataset.map(tokenize_and_encode, batched=True)
    #dataset = tf_dataset.with_format('tensorflow')
    val_dataset.save_to_disk(join('data','book_val'))


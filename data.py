from datasets import  load_dataset# , save_to_disk , load_from_disk
from transformers import BertTokenizer
from os.path import join



tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_and_encode(examples):
    # Tokenize and encode sequences in the dataset
    encoded = tokenizer(examples['text'], return_tensors='tf', truncation=True, padding='max_length', max_length=128, return_special_tokens_mask=True)

    # TensorFlow expects a 'dict' of inputs
    inputs = {'input_ids': encoded['input_ids'], 'token_type_ids': encoded['token_type_ids'], 'attention_mask': encoded['attention_mask']}
    return inputs

if __name__=="__main__":
    dataset = load_dataset("bookcorpus", split="train[:100]")
    # Apply the function to the whole dataset
    tf_dataset = dataset.map(tokenize_and_encode, batched=True)
    tf_dataset = tf_dataset.with_format('tensorflow')
    tf_dataset.save_to_disk(join('data','book_tf'))

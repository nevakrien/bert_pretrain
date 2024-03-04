from transformers import BertConfig, TFBertForMaskedLM

# Define the configuration
config = BertConfig(
    vocab_size=30522,  # Size of the vocabulary, adjust as needed
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    pad_token_id=0,
    position_embedding_type="absolute"
)

# Instantiate the model from the configuration
def get_model():
    return TFBertForMaskedLM(config=config)

if __name__=="__main__":
    model=get_model()
    model.build()
    model.summary()

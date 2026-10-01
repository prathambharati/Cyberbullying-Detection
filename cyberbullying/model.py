"""The network. Training and inference both build it from here, so the saved
weights always match the code that loads them.

One BiLSTM reads the text and feeds two small heads:

* harm: the chance the text is abusive (insults, threats, hate and so on)
* kind: what the abuse is about (age, ethnicity, gender, religion or other)
"""


def build_model(vocab_size: int, max_len: int, num_kinds: int, embedding_dim: int = 100, lstm_units: int = 64):
    import keras
    from keras import layers

    token_ids = keras.Input(shape=(max_len,), dtype="int32", name="token_ids")
    # mask_zero tells the LSTM to skip the padding after short posts.
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="embedding")(token_ids)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units), name="bilstm")(x)
    x = layers.Dropout(0.3)(x)
    outputs = {
        "harm": layers.Dense(1, activation="sigmoid", name="harm")(x),
        "kind": layers.Dense(num_kinds, activation="softmax", name="kind")(x),
    }
    return keras.Model(token_ids, outputs, name="cyberbullying_bilstm")

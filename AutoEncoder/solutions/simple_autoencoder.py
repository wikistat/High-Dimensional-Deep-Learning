autoencoder = km.Sequential(name = "simple_autoencoder")
autoencoder.add(kl.Dense(n_latent, activation='relu', input_shape=(n_input,),name="encoder_layer"))
autoencoder.add(kl.Dense(n_input, activation='sigmoid', name = "decoder_layer" ))
autoencoder.summary()
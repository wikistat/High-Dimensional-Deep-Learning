# encoder
inputs = kl.Input(shape=(784,), name='encoder_input')
x = kl.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = kl.Dense(latent_dim, name='z_mean')(x)
z_log_var = kl.Dense(latent_dim, name='z_log_var')(x)
z = kl.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder_ad = km.Model(inputs, z, name='encoder')

# decoder
latent_inputs = kl.Input(shape=(latent_dim,), name='z_sampling')
x = kl.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = kl.Dense(784, activation='sigmoid')(x)
decoder_ad = km.Model(latent_inputs, outputs, name='decoder')

# vae
outputs = decoder_ad(encoder_ad(inputs))
vae_ad = km.Model(inputs, outputs, name='vae_mlp')

reconstruction_loss = kloss.binary_crossentropy(inputs,outputs)
reconstruction_loss *= 784
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae_ad.add_loss(vae_loss)
vae_ad.compile(optimizer='adam')

vae_ad.fit(x_train_normal,epochs=epochs, batch_size=batch_size, validation_data=(x_test_normal, None))
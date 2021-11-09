# MLP on unidimensional data
n_hidden = 32
model_base_mlp_u =km.Sequential()
model_base_mlp_u.add(kl.Dense(n_hidden, input_shape=(n_features,),  activation = "relu"))
model_base_mlp_u.add(kl.Dropout(0.5))
model_base_mlp_u.add(kl.Dense(n_classes, activation='softmax'))
model_base_mlp_u.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_base_mlp_u.summary()

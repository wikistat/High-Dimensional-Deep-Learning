x_test_normal_decoded = vae_ad.predict(x_test_normal, batch_size=batch_size)
error_normal = np.linalg.norm(x_test_normal-x_test_normal_decoded, axis=1)

x_test_outliers_decoded = vae_ad.predict(x_test_outliers, batch_size =batch_size)
error_outliers = np.linalg.norm(x_test_outliers-x_test_outliers_decoded, axis=1)

x_random = np.random.uniform(size=(1000, 784),low=0.0, high=1.0)
x_random_decoded =  vae_ad.predict(x_random, batch_size=batch_size)
error_random = np.linalg.norm(x_random-x_random_decoded, axis=1)

import CNN
import image_processing as ip
import score_models as sm

model = CNN.vgg_16(img_size=50)

X_test, y_test = CNN.fit_model_batches('arr_X_50_397326', model = model,
                                    weights_filename='50_full_vgg16_batchfit')

probs, cats = CNN.predict_model(X_test, model)

np.save('50_full_vgg16_probs.npy', probs)
np.save('50_full_vgg16_probs.npy', cats)

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

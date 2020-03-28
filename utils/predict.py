import numpy as np

def predict_mask(model, frame_preprocessed):
    pr = model.predict(np.expand_dims(frame_preprocessed, axis=0))[0]
    pr = pr.reshape((model.input_shape[1], model.input_shape[2], 2)).argmax(axis=2)
    return pr
"""Utility function for using Keras."""

def save_keras_model(save_model, path):
    """Save keras model to given path."""
    save_model.save_weights(path + 'model.h5')

    with open(path + 'model.json', "w") as text_file:
        text_file.write(save_model.to_json())

    print('Keras json model saved. {}model.json'.format(path))
    print('Keras h5 model saved. {}model.h5'.format(path))

import pickle

def save_model(network, file_path):
    with open(file_path, 'wb') as f:
        # Use pickle to save the entire network object
        pickle.dump(network, f)

def load_model(file_path):
    with open(file_path, 'rb') as f:
        # Use pickle to load the entire network object
        network = pickle.load(f)
    return network

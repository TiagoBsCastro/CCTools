import tensorflow as tf
import numpy as np
from cosmopower import cosmopower_NN as CPEmulator
from cctools.cctools import get_sigma_r

device = 'CPU'
train = False
test = True

# Define the parameter ranges: [min, max]
param_ranges = {
    'h': [0.6, 0.8],
    'Omega_b': [0.03, 0.06],
    'Omega_c': [0.1, 0.45],
    'A_s': [1.5e-9, 2.5e-9],
    'n_s': [0.9, 1.0],
    'mnu': [0.0, 0.2],       # Neutrino mass
    'Omega_Lambda': [0.6, 0.75], # Omega_Lambda
    'redshift': [0.0, 3.0]   # Redshift
}
parameters = list(param_ranges.keys())

n_test = 5

if(train):

    data =  np.load("../dataprep/data.npz")
    params_array = data['params']
    params_array = params_array.reshape(params_array.shape[0]*params_array.shape[1], params_array.shape[2])

    training_parameters = {}
    for i, k in enumerate(parameters):

        training_parameters[k] = params_array[:, i]

    R_values = data['R_values']
    sigma_R_array = data['sigma_R_array']
    # Initialize the emulator
    emulator = CPEmulator(parameters=parameters,
                          modes=R_values,
                          n_hidden = [512, 512, 512, 512], # 4 hidden layers, each with 512 nodes
                          verbose=True, # useful to understand the different steps in initialisation and training
    )

    # Train the emulator
    with tf.device(device):
        # train
        emulator.train(training_parameters=training_parameters,
                       training_features=sigma_R_array,
                       filename_saved_model='sigmaR',
                       # cooling schedule
                       validation_split=0.1,
                       learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                       batch_sizes=[1024, 1024, 1024, 1024, 1024],
                       gradient_accumulation_steps = [1, 1, 1, 1, 1],
                       # early stopping set up
                       patience_values = [100,100,100,100,100],
                       max_epochs = [1000,1000,1000,1000,1000],
                    )

if(test):

    import matplotlib.pyplot as plt

    data =  np.load("../dataprep/data.npz")
    test_array = data['params'][:n_test]
    test_array = test_array.reshape(test_array.shape[0]*test_array.shape[1], test_array.shape[2])
    R_values = data['R_values']
    sigma_R_array = data['sigma_R_array']

    power_spectra = data['power_spectra'][:n_test]
    sigma_R = get_sigma_r(power_spectra, R_values)

    testing_parameters = {}
    for i, key in enumerate(parameters):
        testing_parameters[key] = np.array(test_array[:, i])

    # Predict sigma(R) for the test set
    emulator = CPEmulator(restore=True,
                          restore_filename='sigmaR',
                          )
    predicted_sigma_R = emulator.predictions_np(testing_parameters)

    # Print the results
    print(predicted_sigma_R)

    # Optionally, plot the predicted sigma(R) values
    for i, (pred, test) in enumerate(zip(predicted_sigma_R, sigma_R)):
        plt.plot(R_values, (pred/test-1)*100, c=f"C{i}")
        #plt.plot(R_values, test, c=f"C{i}")
        #plt.plot(R_values, pred, c=f"C{i}", ls=":")

    plt.xscale('log')
    plt.xlabel('R')
    plt.ylabel('Residual sigma(R) [%]')
    plt.title('Predicted sigma(R)')
    plt.show()

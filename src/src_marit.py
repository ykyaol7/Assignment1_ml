import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def load_data(short=None):
    """
    Load amp data from file
    :param short: (boolean) whether to load a shorter version of the amp data for testing
    :return: (ndarray), array of input data of shape (num_data,)
    """
    if short:
        raw_data = np.load("G:\\mlpr\\assignment1\\raw_data\\short_amp_data2.npy")
    else:
        raw_data = scipy.io.loadmat("G:\\mlpr\\assignment1\\raw_data\\amp_data")['amp_data']
    return np.squeeze(raw_data)


def plot_raw_amp_data(amp_data):
    """
    Plot raw amp data in line graph and histogram
    :param amp_data: (ndarray) Array of amplitude data input
    :return:
    """

    # Set up line graph ploty
    amp_lg = plt.figure("Amplitude Data - Line graph")
    amp_lg.suptitle("Amplitude Data Over Time")
    plt.plot(amp_data)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # switch off tick marks on x axis
    plt.xticks([])

    # Set up histogram plot
    amp_hist = plt.figure("Amplitude Data - Histogram")
    amp_hist.suptitle("Amplitude Data Histogram")
    plt.hist(amp_data, range=(-0.2, 0.2), bins=50, density=True)
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")

    amp_hist = plt.figure("Amplitude Data - Histogram - Full Range")
    amp_hist.suptitle("Amplitude Data Histogram")
    plt.hist(amp_data, bins=50, density=True)
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")

    # Show plots
    plt.show()

    return


##########################################################################################
# QUESTION Would it be better to change the array itself, rather than create a new one? ##
##########################################################################################
def truncate_and_reshape_to_fixed_columns(data_vector, target_cols):
    """
    Creates an array with specified number of columns from an input array, discarding any excess data
    :param data_vector: (ndarray) Array of original data
    :param target_cols: (int) target number of columns
    :return: (ndarray) Array of data with num_cols columns
    """

    # Ensures input is an ndarray with 1 dimension
    data_vector = np.squeeze(data_vector)

    # Return empty array if input array is empty
    if data_vector.shape == (0,):
        return np.array([])

    rows = int(data_vector.shape[0] / target_cols)
    excess = data_vector.shape[0] - rows * target_cols
    if excess == 0:
        return np.reshape(data_vector, newshape=(rows, target_cols))
    return np.reshape(data_vector[:-excess], newshape=(rows, target_cols))


def phi_linear(xx):
    """
    Create a linear design matrix with added bias column
    :param xx: (ndarray) input features
    :return: (ndarray) design matrix with shape (rows in xx, cols in xx + 1)
    """
    # Convert xx is an ndarray
    xx = np.array(xx)

    # If xx is row vector, convert to col vector
    if xx.shape == (xx.shape[0],):
        xx = xx[:, None]

    return np.concatenate([np.ones((xx.shape[0], 1)), xx], axis=1)


def phi_quartic(xx):
    """
    Create a quadratic design matrix with added bias column
    :param xx: (ndarray) input features
    :return: (ndarray) design matrix with shape (rows in xx, 4*(cols in xx) + 1)
    """
    # Convert xx is an ndarray
    xx = np.array(xx)

    # If xx is row vector, convert to col vector
    if xx.shape == (xx.shape[0],):
        xx = xx[:, None]

    return np.concatenate([np.ones((xx.shape[0], 1)), xx, xx**2, xx**3, xx**4], axis=1)


def question1(shuffle=True, save=False):
    """
    Question 1, Assignment 1
    Load and plot raw amplitude data
    Shuffle rows
    Create six arrays
        - X_shuf_train: features for training set
        - X_shuf_val: features for validation set
        - X_shuf_test: features for test set
        - Y_shuf_train: targets for training set
        - Y_shuf_val: targets for validation set
        - Y_shuf_test: targets for test set
    :param shuffle: (bool) Whether to shuffle rows or not.
    :return:
    """
    amp_data = load_data(short=False)

    plot_raw_amp_data(amp_data)
    amp_data = truncate_and_reshape_to_fixed_columns(amp_data, 21)

    if shuffle:
        np.random.seed(1)
        amp_data = np.random.permutation(amp_data)

    # save shuffled data to file
    if save:
        np.save('G:\\mlpr\\assignment1\\raw_data\\shuffled.npy', arr=amp_data)


    # The remaining code in this module feels like a messy way of slicing the data.
    # Surely there must be a neater way of doing this?
    num_rows = amp_data.shape[0]
    train_slice = slice(None, int(num_rows * 0.7))
    val_slice = slice(int(num_rows * 0.7), int(num_rows * (0.7 + 0.15)))
    test_slice = slice(int(num_rows * (0.7 + 0.15)), None)

    # Take first 20 columns as input data
    X_shuf_train = amp_data[train_slice][:, 0:-1]
    X_shuf_val = amp_data[val_slice][:, 0:-1]
    X_shuf_test = amp_data[test_slice][:, 0:-1]

    # Take last column as target data
    Y_shuf_train = amp_data[train_slice][:, -1]
    Y_shuf_val = amp_data[val_slice][:, -1]
    Y_shuf_test = amp_data[test_slice][:, -1]

    return X_shuf_train, X_shuf_val, X_shuf_test, Y_shuf_train, Y_shuf_val, Y_shuf_test


def question2(targets, prediction):
    """
    Question 2, Assignment 1
    Fit linear and quartic curves of amplitude against time through 20 points
    Calculate prediction one step into the future
    :param targets: (ndarray) Target values for 20 input times
    :param prediction: (ndarray) target value for prediction
    :return:
    """
    # Create input data (times)
    times = np.arange(0, 20/20, 1/20)
    row = 0

    # Set up plot
    prediction_plot = plt.figure("Prediction For One Row of Amplitudes")
    prediction_plot.suptitle("Linear and Quartic Fit - Amplitude Prediction")

    # Plot target values
    plt.plot(times, targets[row], 'x', label="Targets - Training data")
    plt.plot(np.array([20/20]), prediction[row], 'rx', label="Target - Prediction")

    # time grid to use for plotting
    times_grid = np.arange(0, 21 / 20, 1 / 20)

    # Calculate and plot linear fit
    w_fit_linear = np.linalg.lstsq(phi_linear(times), targets[row][:, None], rcond=None)[0]
    plt.plot(times_grid, np.dot(phi_linear(times_grid), w_fit_linear), 'b-', label='Linear')

    w_fit_quadratic = np.linalg.lstsq(phi_quartic(times), targets[row][:, None], rcond=None)[0]
    plt.plot(times_grid, np.dot(phi_quartic(times_grid), w_fit_quadratic), 'g-', label='Quartic')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    xx_shuf_train, xx_shuf_val, xx_shuf_test, yy_shuf_train, yy_shuf_val, yy_shuf_test = question1(shuffle=True)
    question2(xx_shuf_train, yy_shuf_train)
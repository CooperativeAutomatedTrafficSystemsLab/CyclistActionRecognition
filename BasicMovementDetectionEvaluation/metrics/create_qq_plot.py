from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt


## This function plots qq plots.
# Normalization can be applied by setting 'normalize=True'.
# @param y_true (np.array): labels
# @param y_pred (np.array): prediction
# @param n_nbins (int): number of bins to plot
# @param title (str): title of figure
def create_qq_plot(y_true, y_pred, n_bins=10, title=''):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, n_bins=n_bins)
    fraction_of_positives = fraction_of_positives.tolist()
    mean_predicted_value = mean_predicted_value.tolist()
    fraction_of_positives.append(1.0)
    fraction_of_positives = [0.0] + fraction_of_positives
    mean_predicted_value.append(1.0)
    mean_predicted_value = [0.0] + mean_predicted_value

    plt.figure()
    plt.title(title)
    plt.plot([0.0, 1.0], [0.0, 1.0], '--')
    plt.plot(mean_predicted_value, fraction_of_positives,
             label="%s" % ('test',))

    return mean_predicted_value, fraction_of_positives

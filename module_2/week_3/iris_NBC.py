import numpy as np


def upload_data(file_path):
    '''
    This function uploads the data from a txt file.
    Args
        file_path: A string of the file path.
    Returns
        data: A numpy array of the data.
    '''
    data = np.genfromtxt(file_path, delimiter=',', dtype='str')
    return np.array(data)


def gauss(x, mean, std):
    '''
    This function computes the Gaussian probability.
    Args
        x: A float of the input value.
        mean: A float of the mean of the Gaussian distribution.
        std: A float of the standard deviation of the Gaussian distribution.
    Returns
        probability: A float of the Gaussian probability.
    '''
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))


def compute_prior_probability(train_data):
    '''
    This function computes the prior probability of the target variable.
    Args
        train_data: A numpy array of the training data.
    Returns
        prior_probability: A numpy array of the prior probability of the target variable.
    '''
    y_unique = np.unique(train_data[:, -1])
    prior_probability = np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        prior_probability[i] = len(np.nonzero(
            train_data[:, -1] == y_unique[i])[0]) / len(train_data)
    return prior_probability


def compute_conditional_probability(train_data):
    '''
    This function computes the conditional probability of the features given the target variable.
    Args
        train_data: A numpy array of the training data.
    Returns
        conditional_probability: A list of the conditional probability of the features given the target variable.
        list_x_name: A list of the unique values of the features.
    '''
    y_unique = np.unique(train_data[:, -1])
    conditional_probability = []

    for i in range(train_data.shape[1]-1):
        x_conditional_probability = np.zeros((len(y_unique), 2))
        for j in range(len(y_unique)):
            mean = np.mean(train_data[np.nonzero(
                train_data[:, -1] == y_unique[j])][:, i].astype(float))
            std = np.std(train_data[np.nonzero(
                train_data[:, -1] == y_unique[j])][:, i].astype(float))
            x_conditional_probability[j][0] = mean
            x_conditional_probability[j][1] = std
        conditional_probability.append(x_conditional_probability)
    return conditional_probability


def train_gausian_naive_bayes(train_data):
    '''
    This function trains a Gaussian Naive Bayes classifier.
    Args
        train_data: A numpy array of the training data.
    Returns
        prior_probability: A numpy array of the prior probability of the target variable.
        conditional_probability: A list of the conditional probability of the features given the target variable.
    '''
    prior_probability = compute_prior_probability(train_data)
    conditional_probability = compute_conditional_probability(train_data)
    return prior_probability, conditional_probability


def predict_gaussian_naive_bayes(conditions, prior_probability, conditional_probability):
    '''
    This function predicts the target variable given the conditions.
    Args
        conditions: A list of the conditions.
        prior_probability: A numpy array of the prior probability of the target variable.
        conditional_probability: A list of the conditional probability of the features given the target variable.
    Returns
        prediction: A string of the predicted target variable.
    '''
    y_unique = np.unique(train_data[:, -1])
    posterior_probability = np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        posterior_probability[i] = prior_probability[i]
        for j in range(len(conditions)):
            posterior_probability[i] *= gauss(
                conditions[j], conditional_probability[j][i][0], conditional_probability[j][i][1])
    prediction = y_unique[np.argmax(posterior_probability)]
    return prediction


if __name__ == '__main__':
    train_data = upload_data('D:/AIO2024/module_2/week_3/iris.data.txt')
    conditions = [6.3, 3.3, 6.0, 2.5]
    prior_probability, conditional_probability = train_gausian_naive_bayes(
        train_data)
    prediction = predict_gaussian_naive_bayes(
        conditions, prior_probability, conditional_probability)
    print(prediction)

import numpy as np


def create_train_data():
    '''
    This function creates a dataset for the play tennis example.
    '''
    data = [['Sunny', 'Hot', 'High', 'Weak', 'no'],
            ['Sunny', 'Hot', 'High', 'Strong', 'no'],
            ['Overcast', 'Hot', 'High', 'Weak', 'yes'],
            ['Rain', 'Mild', 'High', 'Weak', 'yes'],
            ['Rain', 'Cool', 'Normal', 'Weak', 'yes'],
            ['Rain', 'Cool', 'Normal', 'Strong', 'no'],
            ['Overcast', 'Cool', 'Normal', 'Strong', 'yes'],
            ['Overcast', 'Mild', 'High', 'Weak', 'no'],
            ['Sunny', 'Cool', 'Normal', 'Weak', 'yes'],
            ['Rain', 'Mild', 'Normal', 'Weak', 'yes']]
    return np.array(data)


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
    train_data = np.array(train_data)
    y_unique = np.unique(train_data[:, -1])
    conditional_probability = []
    list_x_name = []

    for i in range(0, train_data.shape[1]-1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)
        x_conditional_probability = np.zeros((len(y_unique), len(x_unique)))
        for j in range(len(y_unique)):
            for k in range(len(x_unique)):
                x_conditional_probability[j, k] = len(np.nonzero((train_data[:, i] == x_unique[k]) & (
                    train_data[:, -1] == y_unique[j]))[0]) / len(np.nonzero(train_data[:, -1] == y_unique[j])[0])
        conditional_probability.append(x_conditional_probability)
    return conditional_probability, list_x_name


def get_index_from_value(feature_name, list_features):
    '''
    This function returns the index of a feature in a list of features.
    Args
        feature_name: A string of the feature name.
        list_features: A list of the feature names.
    Returns
        index: An integer of the index of the feature in the list of features.
    '''
    return np.nonzero(list_features == feature_name)[0][0]


def predict_play_tennis(conditions, list_x_name, prior_probability, conditional_probability):
    '''
    This function predicts whether to play tennis or not.
    Args
        condition: A list of the conditions.
        list_x_name: A list of the unique values of the features.
        prior_probability: A numpy array of the prior probability of the target variable.
        conditional_probability: A list of the conditional probability of the features given the target variable.
    Returns
        y_pred: An integer of the predicted target variable.
    '''
    x1 = get_index_from_value(conditions[0], list_x_name[0])
    x2 = get_index_from_value(conditions[1], list_x_name[1])
    x3 = get_index_from_value(conditions[2], list_x_name[2])
    x4 = get_index_from_value(conditions[3], list_x_name[3])

    p0 = prior_probability[0] * conditional_probability[0][0][x1] * conditional_probability[1][0][x2] * \
        conditional_probability[2][0][x3] * conditional_probability[3][0][x4]
    p1 = prior_probability[1] * conditional_probability[0][1][x1] * conditional_probability[1][1][x2] * \
        conditional_probability[2][1][x3] * conditional_probability[3][1][x4]

    if p0 > p1:
        y_pred = 0
    else:
        y_pred = 1

    return y_pred


if __name__ == "__main__":
    train_data = create_train_data()
    prior_probability = compute_prior_probability(train_data)
    print("P(play tennis = 'No')", prior_probability[0])
    print("P(play tennis = 'Yes')", prior_probability[1])

    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)
    print("List_x_name[0]:", list_x_name[0])
    print("List_x_name[1]:", list_x_name[1])
    print("List_x_name[2]:", list_x_name[2])
    print("List_x_name[3]:", list_x_name[3])

    outlook = list_x_name[0]
    i1 = get_index_from_value("Overcast", outlook)
    i2 = get_index_from_value("Rain", outlook)
    i3 = get_index_from_value("Sunny", outlook)
    print(i1, i2, i3)

    x1 = get_index_from_value("Sunny", list_x_name[0])
    print("P('Outlook'='Sunny' | Play Tennis='Yes') = ",
          np.round(conditional_probability[0][1][x1], 2))
    print("P('Outlook'='Sunny' | Play Tennis='No') = ",
          np.round(conditional_probability[0][0][x1], 2))

    X = ['Sunny', 'Cool', 'High', 'Strong']
    pred = predict_play_tennis(
        X, list_x_name, prior_probability, conditional_probability)
    if (pred):
        print('We should go!')
    else:
        print('We should not go!')

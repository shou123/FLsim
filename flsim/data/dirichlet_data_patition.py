import os 
import numpy as np
from torchvision.datasets.cifar import CIFAR10
from keras.datasets import cifar10
import torchvision.transforms as transforms

def save_cifar10_party_data(should_stratify, party_folder):
    """
    Saves Cifar10 party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    # if not os.path.exists(dataset_folder):
    #     os.makedirs(dataset_folder)

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CIFAR10(
        root="../cifar10", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR10(
        root="../cifar10", train=False, download=True, transform=transform
    )

        # Select the top 25,000 data points
    top_25000_indices = np.arange(25000)
    nb_dp_per_party = [train_dataset[i] for i in top_25000_indices]

    x_train, y_train = train_dataset.data, train_dataset.targets
    x_test, y_test = test_dataset.data, test_dataset.targets

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # (x_train, y_train), (x_test, y_test) = partition_data(dataset='cifar10',dataset_folder= dataset_folder, partition="hetero-dir",n_nets=8, alpha=0.5 )
    labels, train_counts = np.unique(y_train, return_counts=True)
    te_labels, test_counts = np.unique(y_test, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")
       

    num_train = np.shape(y_train)[0]
    num_test = np.shape(y_test)[0]
    num_labels = np.shape(np.unique(y_test))[0]
    nb_parties = len(nb_dp_per_party)

    if should_stratify:
        # Sample according to source label distribution
        train_probs = {
            label: train_counts[label] / float(num_train) for label in labels}
        test_probs = {label: test_counts[label] /
                      float(num_test) for label in te_labels}
    else:
        # Sample uniformly
        #dirichlet distribution
        # proportions = np.random.dirichlet(np.repeat(0.5, len(labels)))
        # train_probs = {label: proportions[label] for label in labels}
       
        #uniform dis
        # train_probs = {label: 1.0 / len(labels) for label in labels}

        # test_probs = {label: 1.0 / len(te_labels) for label in te_labels}
       
        #print(proportions)
        #print(train_probs)
        print(num_train)
        # probs = {label: 1.0 / num_train for label in labels}
        # p_list = np.array([probs[y_train[idx]] for idx in range(num_train)])
        # p_list /= np.sum(p_list)
       
    for idx, dp in enumerate(nb_dp_per_party):
        #dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(0.4, len(labels)))
        train_probs = {label: proportions[label] for label in labels}
        #test_probs  = train_probs
        test_probs = train_probs

        print(labels)
        print (te_labels)
        print(train_probs)
        print(test_probs)
        train_p = np.array([train_probs[y_train[idx]]
                            for idx in range(num_train)])
        train_p = np.array(train_p)
        train_p /= np.sum(train_p)
        train_indices = np.random.choice(num_train, dp, p=train_p)
        test_p = np.array([test_probs[y_test[idx]] for idx in range(num_test)])
        test_p /= np.sum(test_p)

        # Split test evenly
        test_indices = np.random.choice(
            num_test, int(num_test / nb_parties), p=test_p)

        x_train_pi = x_train[train_indices]
        y_train_pi = y_train[train_indices]
        x_test_pi = x_test[test_indices]
        y_test_pi = y_test[test_indices]

        # Now put it all in an npz
        name_file = 'data_party' + str(idx) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
                 x_test=x_test_pi, y_test=y_test_pi)

        # print_statistics(idx, x_test_pi, x_train_pi, num_labels, y_train_pi)

        for l in range(num_labels):
            print('* Test Label ', l, ' samples: ', (y_test_pi == l).sum())

        print('Finished! :) Data saved in ', party_folder)


if __name__ == "__main__":
    save_cifar10_party_data(True, "/home/shiyue/FLsim/dirichlet_data_partition")
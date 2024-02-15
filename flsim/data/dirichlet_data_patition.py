import os
import numpy as np
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
import torch

# def save_cifar10_party_data(should_stratify, party_folder):
def save_cifar10_party_data(client_number,sample_per_user,dirichlet_alph,data_type= "iid"):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CIFAR10(
        root="/home/shiyue/FLsim/cifar10", train=True, download=False, transform=transform
    )
    test_dataset = CIFAR10(
        root="/home/shiyue/FLsim/cifar10", train=False, download=False, transform=transform
    )

    client_number = client_number
    nb_dp_per_party = [sample_per_user] * client_number

    x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
    x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

    labels, train_counts = np.unique(y_train, return_counts=True)
    te_labels, test_counts = np.unique(y_test, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")

    num_train = np.shape(y_train)[0]
    num_test = np.shape(y_test)[0]
    num_labels = np.shape(np.unique(y_test))[0]
    nb_parties = len(nb_dp_per_party)

    train_party_data_list = []
    test_party_data_list = []

    
    
    for idx, dp in enumerate(nb_dp_per_party):
        if data_type == "iid":
            equal_probability = 1.0 / len(labels)
            proportions = np.full(len(labels), equal_probability)
        elif data_type == "non_iid":
        #dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(dirichlet_alph, len(labels)))
        
        train_probs = {label: proportions[label] for label in labels}
        #test_probs  = train_probs
        test_probs = train_probs

        print(f"train_labels: {labels}")
        print(f"train_probs: {train_probs}")
        print (f"test_labels: {te_labels}")
        print(f"test_probs: {test_probs}")
        train_p = np.array([train_probs[y_train[idx]]
                            for idx in range(num_train)])
        train_p = np.array(train_p)
        train_p /= np.sum(train_p)
        train_indices = np.random.choice(num_train, dp, p=train_p)
        test_p = np.array([test_probs[y_test[idx]] for idx in range(num_test)])
        test_p /= np.sum(test_p)
        # test_indices = np.random.choice(num_test, int(num_test / nb_parties), p=test_p)
        test_indices = np.random.choice(num_test, int(num_test / len(labels)), p=test_p) #change the test dataset for num_test = 10,000/10 = 1000


        # # Split test evenly
        # test_indices = np.random.choice(
        #     num_test, int(num_test / nb_parties), p=test_p)

        x_train_pi = x_train[train_indices]
        y_train_pi = y_train[train_indices]
        x_test_pi = x_test[test_indices]
        y_test_pi = y_test[test_indices]

        train_tensor_data_normalized = (torch.from_numpy(x_train_pi).float() / 255.0).permute(0,3,1,2)
        test_tensor_data_normalized = (torch.from_numpy(x_test_pi).float() / 255.0).permute(0,3,1,2)

        with open("/home/shiyue/FLsim/results/non-iid_data_distribution.txt", "a") as file:
            for l in range(num_labels):
                print('Client',idx,'* Train Label ', l, ' samples: ', (y_train_pi == l).sum())
                file.write(f'Client {idx}: Train Label {l}, samples: {(y_train_pi == l).sum()}\n')

            for l in range(num_labels):
                print('Client',idx,'* Test Label ', l, ' samples: ', (y_test_pi == l).sum())
                file.write(f'Client {idx}: Test Label {l}, samples: {(y_test_pi == l).sum()}\n')

            # print('Finished! :) Data saved in ', party_folder)

                    # Split test evenly
            train_samples = [{'features': train_tensor_data_normalized[i], 'labels': int(y_train_pi[i])} for i in range(dp)]
            # test_samples = [{'features': test_tensor_data_normalized[i], 'labels': int(y_test_pi[i])} for i in range(int(num_test / nb_parties))]
            test_samples = [{'features': test_tensor_data_normalized[i], 'labels': int(y_test_pi[i])} for i in range(int(num_test / len(labels)))] #change the test dataset for num_test = 10,000/10 = 1000


            train_party_data_list.append({f"{idx:04d}": train_samples})
            test_party_data_list.append({f"{idx:04d}": test_samples})

    return train_party_data_list,test_party_data_list

    

# if __name__ == "__main__":
#     save_cifar10_party_data(10,5000,0.4)
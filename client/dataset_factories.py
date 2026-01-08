import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.datasets import load_svmlight_files
from torch.utils.data import TensorDataset
from .datasets_setup import DATASET_PATHS, download_UCI_dataset
import os

class datasets():

    def __init__(self, dataset, num_clients, params={'data_per_client': 'sequential'}):

        if dataset == 'CIFAR10':

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)

            self.testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)

            self.num_classes = 10

        elif dataset == 'CIFAR100':

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform_train)

            self.testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform_test)

            self.num_classes = 100

        elif dataset == 'MNIST':

            transform_train = transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])

            transform_test = transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.trainset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform_train)

            self.testset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform_test)

            self.num_classes = 10


        elif dataset == 'tinyimagenet':

            # Define main data directory
            DATA_DIR = './data/tiny-imagenet-200'  # Original images come in shapes of [3,64,64]

            # Define training and validation data paths
            TRAIN_DIR = os.path.join(DATA_DIR, 'train')
            VALID_DIR = os.path.join(DATA_DIR, 'val')

            # Create separate validation subfolders for the validation images based on
            # their labels indicated in the val_annotations txt file
            val_img_dir = os.path.join(VALID_DIR, 'images')

            if not os.path.isdir(DATA_DIR): # dataset isn't stored
                os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
                os.system('unzip -qq tiny-imagenet-200.zip -d ./data')
                os.remove('tiny-imagenet-200.zip')

                # Open and read val annotations text file
                with open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r') as fp:
                    data = fp.readlines()

                    # Create dictionary to store img filename (word 0) and corresponding
                    # label (word 1) for every line in the txt file (as key value pair)
                    val_img_dict = {}
                    for line in data:
                        words = line.split('\t')
                        val_img_dict[words[0]] = words[1]

                # Create subfolders (if not present) for validation images based on label,
                # and move images into the respective folders
                for img, folder in val_img_dict.items():
                    newpath = (os.path.join(val_img_dir, folder))
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    if os.path.exists(os.path.join(val_img_dir, img)):
                        os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

            # Define transformation sequence for image pre-processing
            # If not using pre-trained model, normalize with 0.5, 0.5, 0.5 (mean and SD)
            # If using pre-trained ImageNet, normalize with mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225])

            preprocess_transform_pretrain = transforms.Compose([
                transforms.Resize(256),  # Resize images to 256 x 256
                transforms.CenterCrop(224),  # Center crop image
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Converting cropped images to tensors
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

            self.trainset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=preprocess_transform_pretrain)
            self.testset = torchvision.datasets.ImageFolder(val_img_dir, transform=preprocess_transform_pretrain)

            self.num_classes = 200


        elif dataset == 'a4a' or dataset == 'a9a':
            paths = DATASET_PATHS[dataset]
            if not (paths['train'].exists() and paths['test'].exists()):
                download_UCI_dataset(dataset)
            train_x, train_y, test_x, test_y = load_svmlight_files([str(paths['train']), str(paths['test'])])

            train_x = train_x[:-1]
            train_y = train_y[:-1]

            def convert_UCI_to_torch(x, y):
                return (torch.from_numpy(x.toarray().astype(np.float32)),
                        torch.reshape(torch.from_numpy(((y + 1) / 2).astype(np.float32)), [-1, 1]))

            self.trainset = TensorDataset(*convert_UCI_to_torch(train_x, train_y))
            self.testset = TensorDataset(*convert_UCI_to_torch(test_x, test_y))

            self.num_classes = 2

        else:
            raise Exception('Unsupported dataset {}'.format(dataset))

        self.num_clients = num_clients
        self.trainset_len = len(self.trainset)
        self.testset_len = len(self.testset)
        self.params = params
        self.indices = {}

        # --- INSERT THIS FOR ROBUST SHUFFLING ---
        # Pre-compute a global random permutation of indices.
        # This guarantees that 'iid' mode gives purely random subsets to every client.
        if self.params['data_per_client'] == 'iid':
            # Use a fixed seed generator for reproducibility of the split
            generator = torch.Generator()
            generator.manual_seed(42) 
            self.iid_indices = torch.randperm(self.trainset_len, generator=generator).tolist()
        # ----------------------------------------

    def get_client_train_data(self, client_id):

        if self.params['data_per_client'] == 'sequential':

            # assume that self.trainset_len >> self.num_clients
            k, m = divmod(self.trainset_len, self.num_clients)
            indices = list(range(client_id * k + min(client_id, m), (client_id + 1) * k + min(client_id + 1, m)))

            # debug
            subset = torch.utils.data.Subset(self.trainset, indices)
            print('(train) client id {} ({})'.format(client_id, len(subset.indices)))

            return torch.utils.data.Subset(self.trainset, indices)

# ============================================================
        # NEW: IID Distribution Mode
        # ============================================================
        if self.params['data_per_client'] == 'iid':
            # Assign a contiguous chunk of the *randomly permuted* indices to this client.
            # This ensures Client 0 gets random data, Client 1 gets random data, etc.
            
            k, m = divmod(self.trainset_len, self.num_clients)
            
            # Calculate start/end pointers in the permuted list
            start_idx = client_id * k + min(client_id, m)
            end_idx = (client_id + 1) * k + min(client_id + 1, m)
            
            # Extract the random indices for this client
            client_indices = self.iid_indices[start_idx : end_idx]

            # Create the subset
            subset = torch.utils.data.Subset(self.trainset, client_indices)
            print('(train) client id {} assigned {} random IID samples'.format(client_id, len(subset.indices)))

            return subset

        elif self.params['data_per_client'] == 'label_per_client':

            # Assuming that classes start from 0, similarly to client_id. Holds for CIFAR10 and CIFAR100. ### TODO: why not for MNIST?
            indices = np.where(np.array(self.trainset.targets) == (client_id % self.num_classes))[0]

            # debug
            subset = torch.utils.data.Subset(self.trainset, indices.tolist())
            print('(train) client id {} with {} label ({})'.format(client_id, client_id % self.num_classes, len(subset.indices)))

            unique_targets = np.unique([subset.dataset[idx][1] for idx in subset.indices])
            if not (len(unique_targets) == 1 and unique_targets[0] == client_id % self.num_classes):
                raise Exception('Error, label_per_client (train) misbehaves, expecting class {} for client {}, '
                                'but getting class(es) {}'.format(client_id % self.num_classes, client_id,
                                                                  unique_targets))

            return torch.utils.data.Subset(self.trainset, indices.tolist())

        elif self.params['data_per_client'] == 'label_per_client1':
            iid_trainset, per_label_trainset = torch.utils.data.random_split(self.trainset, [int(0.2*len(self.trainset)), int(0.8*len(self.trainset))], generator=torch.Generator().manual_seed(42))
            # Assuming that classes start from 0, similarly to client_id. Holds for CIFAR10 and CIFAR100. ### TODO: why not for MNIST?

            # assume that self.trainset_len >> self.num_clients
            k, m = divmod(len(iid_trainset), self.num_clients)
            iid_indices = list(range(client_id * k + min(client_id, m), (client_id + 1) * k + min(client_id + 1, m)))
            per_label_indices = np.where(np.array(per_label_trainset.targets) == (client_id % self.num_classes))[0]

            return torch.utils.data.ConcatDataset(torch.utils.data.Subset(per_label_trainset, per_label_indices.tolist()), torch.utils.data.Subset(iid_trainset, iid_indices))

        else:

            raise Exception('Unknown params[\'data_per_client\']: {}'.format(self.params['data_per_client']))

    def get_client_test_data(self, client_id):

        if self.params['data_per_client'] == 'sequential':

            # assume that self.testset_len >> self.num_clients
            k, m = divmod(self.testset_len, self.num_clients)
            indices = list(range(client_id * k + min(client_id, m), (client_id + 1) * k + min(client_id + 1, m)))
            self.indices[client_id] = indices

            # debug
            subset = torch.utils.data.Subset(self.testset, indices)
            print('(test) client id {} ({})'.format(client_id, len(subset.indices)))


            return torch.utils.data.Subset(self.testset, indices)

        elif self.params['data_per_client'] == 'label_per_client':

            # Assuming that classes start from 0, similarly to client_id. Holds for CIFAR10 and CIFAR100.
            indices = np.where(np.array(self.testset.targets) == (client_id % self.num_classes))[0]
            self.indices[client_id] = indices
            # debug
            subset = torch.utils.data.Subset(self.testset, indices.tolist())
            print('(test) client id {} with {} label ({})'.format(client_id, client_id % self.num_classes, len(subset.indices)))

            unique_targets = np.unique([subset.dataset[idx][1] for idx in subset.indices])
            if not (len(unique_targets) == 1 and unique_targets[0] == client_id % self.num_classes):
                raise Exception('Error, label_per_client (test) misbehaves, expecting class {} for client {}, '
                                'but getting class(es) {}'.format(client_id % self.num_classes, client_id,
                                                                  unique_targets))


            return torch.utils.data.Subset(self.testset, indices.tolist())

        else:

            raise Exception('Unknown params[\'data_per_client\']: {}'.format(self.params['data_per_client']))

    def get_train_data(self):
        return self.trainset

    def get_test_data(self):
        return self.testset

    def get_indices(self):
        return self.indices
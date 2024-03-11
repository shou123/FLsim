#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In this tutorial, we will train an image classifier with FLSim to simulate a federated learning training environment.

With this tutorial, you will learn the following key components of FLSim:
1. Data loading
2. Model construction
3. Trainer construction

    Typical usage example:
    python3 cifar10_example.py --config-file configs/cifar10_config.json
"""
import flsim.configs  # noqa
import hydra
import torch
from flsim.data.data_sharder import SequentialSharder,RandomSharder,PowerLawSharder
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils import (
    DataLoader,
    DataProvider,
    FLModel,
    MetricsReporter,
    SimpleConvNet,
    DataLoaderForNonIID,
    Resnet18,
    Test_Net
)

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
import numpy as np
from flsim.data.dirichlet_data_patition import save_cifar10_party_data
from torchvision.datasets.cifar import CIFAR10
# from torchvision.models import resnet18
## try to load local resnet18


IMAGE_SIZE = 32
intermediate_test_data = None


def build_data_provider(local_batch_size, examples_per_user,data_type,dirichlet_alph, drop_last: bool = False, total_client_num = 0):

    #============================================iid===============================================================
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize(IMAGE_SIZE),
    #         transforms.CenterCrop(IMAGE_SIZE),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ]
    # )
    # transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # train_dataset = CIFAR10(
    #     root="/home/shiyue/FLsim/cifar10", train=True, download=True, transform=transform_train
    # )
    # test_dataset = CIFAR10(
    #     root="/home/shiyue/FLsim/cifar10", train=False, download=True, transform=transform_test
    # )

    # sharder = SequentialSharder(examples_per_shard=examples_per_user)
    # # sharder = RandomSharder(num_shards=10)
    # # sharder = PowerLawSharder(num_shards=10,alpha = 0.8)

    # fl_data_loader = DataLoader(train_dataset, test_dataset, test_dataset, sharder, local_batch_size, drop_last)
    #================================================================================================================

    #============================================== non iid=====================================================================
    total_client_num = total_client_num
    # data_type = "non_iid"
    # data_type = "iid"
    # dirichlet_alph = 0.4
    train_party_data_list,test_party_data_list = save_cifar10_party_data(total_client_num,examples_per_user,dirichlet_alph = dirichlet_alph,data_type = data_type)
    global intermediate_test_data
    intermediate_test_data = test_party_data_list
    sharder = SequentialSharder(examples_per_shard=examples_per_user)
    fl_data_loader = DataLoaderForNonIID(train_party_data_list, test_party_data_list, test_party_data_list, sharder, local_batch_size, drop_last)
    #===========================================================================================================================
    

    data_provider = DataProvider(fl_data_loader)
    return data_provider


def main(
    trainer_config,
    data_config,
    use_cuda_if_available: bool = True,
) -> None:

    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    # model = SimpleConvNet(in_channels=3, num_classes=10) #This is my original model
    # model = Resnet18(num_classes=10)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model = Test_Net() # This model is CNN from Allen

    # define the data sample type
    data_type = "non_iid"
    # data_type = "iid"

    #define non iid data dirichlet_alph value 
    dirichlet_alph = 0.4

    #set for largest_distance_select_persentage
    largest_distance_select_persentage = 0.4 


    # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()
    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    data_provider = build_data_provider(
        local_batch_size=data_config.local_batch_size,
        examples_per_user=data_config.examples_per_user,
        total_client_num = data_config.total_client_num,
        # examples_per_user = trainer_config.users_per_round,
        drop_last=False,
        data_type = data_type,
        dirichlet_alph = dirichlet_alph,
    )

    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=1,
        evaluate_data = intermediate_test_data,
        largest_distance_select_percentage = largest_distance_select_persentage
    )

    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter([Channel.STDOUT]),
    )


@hydra.main(config_path=None, config_name="cifar10_tutorial")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    data_config = cfg.data

    main(
        trainer_config,
        data_config,
    )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)




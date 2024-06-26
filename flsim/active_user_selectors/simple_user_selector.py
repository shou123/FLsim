#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from flsim.common.pytest_helper import assertNotEmpty
from flsim.data.data_provider import IFLDataProvider
from flsim.utils.config_utils import fullclassname, init_self_cfg
from omegaconf import MISSING
import numpy as np

import random
import re





class ActiveUserSelectorUtils:
    @staticmethod
    def convert_to_probability(
        user_utility: torch.Tensor,
        fraction_with_zero_prob: float,
        softmax_temperature: float,
        weights=None,
    ) -> torch.Tensor:
        if weights is None:
            weights = torch.ones(len(user_utility), dtype=torch.float)
        num_to_zero_out = math.floor(fraction_with_zero_prob * len(user_utility))

        sorted_indices = torch.argsort(user_utility, descending=True).tolist()
        unnormalized_probs = torch.exp(softmax_temperature * user_utility) * weights
        if num_to_zero_out > 0:
            for i in sorted_indices[-num_to_zero_out:]:
                unnormalized_probs[i] = 0

        tmp_sum = sum(unnormalized_probs.tolist())
        assert tmp_sum > 0
        normalized_probs = unnormalized_probs / tmp_sum

        return normalized_probs

    @staticmethod
    def normalize_by_sample_count(
        user_utility: torch.Tensor,
        user_sample_counts: torch.Tensor,
        averaging_exponent: float,
    ) -> torch.Tensor:
        # pyre-fixme[58]: `/` is not supported for operand types `int` and `Tensor`.
        sample_averaging_weights = 1 / torch.pow(user_sample_counts, averaging_exponent)
        user_utility = sample_averaging_weights * user_utility
        return user_utility

    @staticmethod
    def samples_per_user(data_provider: IFLDataProvider) -> torch.Tensor:
        samples_per_user = [
            data_provider.get_train_user(u).num_train_examples()
            for u in data_provider.train_user_ids()
        ]
        samples_per_user = torch.tensor(samples_per_user, dtype=torch.float)
        return samples_per_user

    @staticmethod
    def select_users(
        users_per_round: int,
        probs: torch.Tensor,
        fraction_uniformly_random: float,
        rng: Any,
    ) -> List[int]:
        num_total_users = len(probs)
        num_randomly_selected = math.floor(users_per_round * fraction_uniformly_random)
        num_actively_selected = users_per_round - num_randomly_selected
        assert len(torch.nonzero(probs)) >= num_actively_selected

        if num_actively_selected > 0:

            actively_selected_indices = torch.multinomial(
                probs, num_actively_selected, replacement=False, generator=rng
            ).tolist()
        else:
            actively_selected_indices = []

        if num_randomly_selected > 0:
            tmp_probs = torch.tensor(
                [
                    0 if x in actively_selected_indices else 1
                    for x in range(num_total_users)
                ],
                dtype=torch.float,
            )

            randomly_selected_indices = torch.multinomial(
                tmp_probs, num_randomly_selected, replacement=False, generator=rng
            ).tolist()
        else:
            randomly_selected_indices = []

        selected_indices = actively_selected_indices + randomly_selected_indices
        return selected_indices

    @staticmethod
    def sample_available_users(
        users_per_round: int, available_users: List[int], rng: torch.Generator
    ) -> List[int]:
        if users_per_round >= len(available_users):
            return copy.copy(available_users)

        selected_indices = torch.multinomial(
            torch.ones(len(available_users), dtype=torch.float),
            users_per_round,
            replacement=False,
            generator=rng,
        ).tolist()

        return [available_users[idx] for idx in selected_indices]


class ActiveUserSelector(abc.ABC):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ActiveUserSelectorConfig,
            **kwargs,
        )

        self.rng = torch.Generator()
        if self.cfg.user_selector_seed is not None:
            self.rng = self.rng.manual_seed(self.cfg.user_selector_seed)
        else:
            self.rng.seed()

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @abc.abstractmethod
    def get_user_indices(self, **kwargs) -> List[int]:
        pass

    def get_users_unif_rand(
        self, num_total_users: int, users_per_round: int
    ) -> List[int]:
        selected_indices = torch.multinomial(
            torch.ones(num_total_users, dtype=torch.float),
            users_per_round,
            replacement=False,
            generator=self.rng,
        ).tolist()

        return selected_indices

    def unpack_required_inputs(
        self, required_inputs: List[str], kwargs: Dict[str, Any]
    ) -> List[Any]:
        inputs = []
        for key in required_inputs:
            if key == "num_samples_per_user":
                array_length = kwargs.get("num_total_users",None)
                input = [5000]*array_length #sample size per user

            else:
                input = kwargs.get(key, None)
            assert (
                input is not None
            ), "Input `{}` is required for get_user_indices in active_user_selector {}.".format(
                key, self.__class__.__name__
            )
            inputs.append(input)
        return inputs


class UniformlyRandomActiveUserSelector(ActiveUserSelector):
    """Simple User Selector which does random sampling of users"""

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=UniformlyRandomActiveUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round"]
        num_total_users, users_per_round = self.unpack_required_inputs(
            required_inputs, kwargs
        )

        selected_indices = torch.multinomial(
            torch.ones(num_total_users, dtype=torch.float),
            users_per_round,
            # pyre-fixme[16]: `UniformlyRandomActiveUserSelector` has no attribute
            #  `cfg`.
            replacement=self.cfg.random_with_replacement,
            generator=self.rng,
        ).tolist()

        print(f"client index: {selected_indices}")
        return selected_indices


class SequentialActiveUserSelector(ActiveUserSelector):
    """Simple User Selector which chooses users in sequential manner.
    e.g. if 2 users (user0 and user1) were trained in the previous round,
    the next 2 users (user2 and user3) will be picked in the current round.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=SequentialActiveUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)
        self.cur_round_user_index = 0

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round"]
        num_total_users, users_per_round = self.unpack_required_inputs(
            required_inputs, kwargs
        )

        # when having covered all the users, return the cursor to 0
        if num_total_users <= self.cur_round_user_index:
            self.cur_round_user_index = 0

        next_round_user_index = self.cur_round_user_index + users_per_round
        user_indices = list(
            range(
                self.cur_round_user_index, min(next_round_user_index, num_total_users)
            )
        )
        self.cur_round_user_index = next_round_user_index
        
        print(f"client index: {user_indices}")
        return user_indices


class RandomRoundRobinActiveUserSelector(ActiveUserSelector):
    """User Selector which chooses users randomly in a round-robin fashion.
    Each round users are selected uniformly randomly from the users not
    yet selected in that epoch.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=RandomRoundRobinActiveUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)
        self.available_users = []

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round"]
        num_total_users, users_per_round = self.unpack_required_inputs(
            required_inputs, kwargs
        )
        # when having covered all the users, reset the list of available users
        if len(self.available_users) == 0:
            self.available_users = list(range(num_total_users))

        user_indices = ActiveUserSelectorUtils.sample_available_users(
            users_per_round, self.available_users, self.rng
        )

        # Update the list of available users
        # TODO(dlazar): ensure this is the fastest method. If not, write a util
        self.available_users = [
            idx for idx in self.available_users if idx not in user_indices
        ]

        print(f"client index: {user_indices}")
        return user_indices


class ImportanceSamplingActiveUserSelector(ActiveUserSelector):
    """User selector which performs Important Sampling.
    Each user is randomly selected with probability =
        `number of samples in user * clients per round / total samples in dataset`
    Ref: https://arxiv.org/pdf/1809.04146.pdf
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ImportanceSamplingActiveUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round", "num_samples_per_user"]
        (
            num_total_users,
            users_per_round,
            num_samples_per_user,
        ) = self.unpack_required_inputs(required_inputs, kwargs)

        assert (
            len(num_samples_per_user) == num_total_users
        ), "Mismatch between num_total_users and num_samples_per_user length"
        assert users_per_round > 0, "users_per_round must be greater than 0"

        prob = torch.tensor(num_samples_per_user).float()
        print(f"Importance Sampling Active prob value: {prob}")

        total_samples = torch.sum(prob)
        print(f"Importance Sampling Active total_samples: {total_samples}")

        assert total_samples > 0, "All clients have empty data"
        prob = prob * users_per_round / total_samples
        print(f"Importance Sampling Active final prob value: {prob}")
        # Iterate num_tries times to ensure that selected indices is non-empty
        selected_indices = []
        # pyre-fixme[16]: `ImportanceSamplingActiveUserSelector` has no attribute `cfg`.
        for _ in range(self.cfg.num_tries):
            selected_indices = (
                torch.nonzero(torch.rand(num_total_users, generator=self.rng) < prob)
                .flatten()
                .tolist()
            )
            if len(selected_indices) > 0:
                break

        assertNotEmpty(
            selected_indices,
            "Importance Sampling did not return any clients for the current round",
        )

        print(f"client index: {selected_indices}")
        return selected_indices


class RandomMultiStepActiveUserSelector(ActiveUserSelector):
    """Simple User Selector which does random sampling of users"""

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=RandomMultiStepActiveUserSelectorConfig,
            **kwargs,
        )
        self.gamma = self.cfg.gamma
        self.milestones = self.cfg.milestones
        self.users_per_round = 0
        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round", "global_round_num"]
        (
            num_total_users,
            users_per_round,
            global_round_num,
        ) = self.unpack_required_inputs(required_inputs, kwargs)

        if global_round_num in self.milestones:
            self.users_per_round *= self.gamma
            print(f"Increase Users Per Round to {self.users_per_round}")
        elif self.users_per_round == 0:
            self.users_per_round = users_per_round

        selected_indices = torch.multinomial(
            torch.ones(num_total_users, dtype=torch.float),
            self.users_per_round,
            # pyre-ignore[16]
            replacement=self.cfg.random_with_replacement,
            generator=self.rng,
        ).tolist()

        print(f"client index: {selected_indices}")
        return selected_indices


class LargestDistanceActiveUserSelector(ActiveUserSelector):
    """According to the distance between global module and local module to select client
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=LargestDistanceActiveUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)
        self.cur_round_user_index = 0

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round","select_percentage","client_local_model","global_model","epoch_num"]
        num_total_users, users_per_round,select_percentage, client_local_model,global_model,epoch_num= self.unpack_required_inputs(
            required_inputs, kwargs
        )

        total_elements = int(select_percentage * num_total_users)
        user_indices = []
        self._user_indices_overselected = []
        #at the first round, if client_local_model bucket is empty, select all client to training and save the ;oca; models
        if len(client_local_model)== 0:
            for i in range (num_total_users):
                user_indices.append(i)
                self._user_indices_overselected = user_indices
        
        else:
            # List to store Frobenius norms and corresponding keys
            clients_distance = []

            # Iterate over each dictionary in self.client_model_deltas
            for client_model in client_local_model:
                for key, value in client_model.items():
                    # Subtract the value from global_model
                    distance = global_model - value
                    frobenius_norm = distance.norm('fro')
                    # Append the Frobenius norm and corresponding key to the list
                    clients_distance.append((key, frobenius_norm.item()))

            # Sort client distance and according to selected percentage to select clients
            sorted_clients_distance = sorted(clients_distance, key=lambda x: x[1], reverse=True)
            with open("results/sorted_largest_client_distance.txt", 'a') as file:
                for client, distance in sorted_clients_distance:
                    client_norm_info = "Global_round: {}, Client: {}, distance: {}\n".format(epoch_num,client, distance)
                    file.write(client_norm_info)

            # Select the top 80% largest values along with their corresponding keys
            for key, _ in sorted_clients_distance[0:total_elements]:
               self._user_indices_overselected.append(key)

    
        print(f"client index: {self._user_indices_overselected}")
        return self._user_indices_overselected


class ModelLayerActiveUserSelector(ActiveUserSelector):
    """Simple User Selector which chooses users in sequential manner.
    e.g. if 2 users (user0 and user1) were trained in the previous round,
    the next 2 users (user2 and user3) will be picked in the current round.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ModelLayerActiveUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)
        self.cur_round_user_index = 0

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round","select_percentage","total_updated_models_instance","global_model_instance","epoch_num", "learnable_layers"]
        num_total_users, users_per_round,select_percentage, total_updated_models_instance,global_model_instance,epoch_num,learnable_layers= self.unpack_required_inputs(
            required_inputs, kwargs
        )

        total_elements = int(select_percentage * num_total_users)
        user_indices = []
        self._user_indices_overselected = []
        # at the first round, if client_local_model bucket is empty, select all client to training and save the ;oca; models
        if len(total_updated_models_instance)== 0:
            for i in range (num_total_users):
                user_indices.append(i)
                self._user_indices_overselected = user_indices
        
        else:
            # List to store Frobenius norms and corresponding keys
            distance = []

            # Iterate over each dictionary in self.client_model_deltas
            for client_id,client_model in enumerate(total_updated_models_instance):
                # distance = []
                learnable_layers_distance = 0
                for layer_index, (client_layer, server_layer) in enumerate(zip(client_model[client_id], global_model_instance.fl_get_module().state_dict())):
                    if 'weight' in client_layer and 'weight' in server_layer:
                        match = re.search(r'\.(\d+)\.', client_layer) #to get the number of 'network.16.weight'
                        if match:
                            number = match.group(1)
                            if int(number) in learnable_layers:
                                client_model_layer_tensor = client_model[client_id][client_layer]
                                global_model_layer_tensor = global_model_instance.fl_get_module().state_dict()[client_layer]

                                client_model_layer_data = client_model_layer_tensor.data.view(-1)
                                global_model_layer_data = global_model_layer_tensor.data.view(-1)
                                layer_distance = torch.norm(client_model_layer_data-global_model_layer_data, 'fro')
                                with open("results/layer_distance.txt", 'a') as file:
                                    file.write(f"epoch_num： {epoch_num}, client_id: {client_id}, layer_id: {match.string}, layer_distance: {layer_distance}\n")
                                print(f"epoch_num： {epoch_num}, client_id: {client_id}, layer_id: {match.string}, layer_distance: {layer_distance}")

                            # # np.linalg.norm with flatten
                            # if int(number) in learnable_layers:
                            #     client_model_layer_tensor = client_model.fl_get_module().state_dict()[client_layer]
                            #     global_model_layer_tensor = global_model_instance.fl_get_module().state_dict()[client_layer]
                            #     client_model_layer_data = client_model_layer_tensor.view(-1).cpu().numpy()  # Convert to numpy array
                            #     global_model_layer_data = global_model_layer_tensor.view(-1).cpu().numpy()  # Convert to numpy array
                            #     linalg_flatten_layer_distance = np.linalg.norm(global_model_layer_data - client_model_layer_data)
                            #     with open("results/linalg_flatten_layer_distance.txt", 'a') as file:
                            #         file.write(f"epoch_num： {epoch_num}, client_id: {client_id}, layer_id: {match.string}, layer_distance: {linalg_flatten_layer_distance}\n")
                                
                            # # np.linalg.norm with unflatten                        
                            # if int(number) in learnable_layers:
                            #     client_model_layer = client_model.fl_get_module().state_dict()[client_layer].cpu().numpy()
                            #     global_model_layer = global_model_instance.fl_get_module().state_dict()[client_layer].cpu().numpy()
                            #     linalg_unflatten_layer_distance = np.linalg.norm(global_model_layer - client_model_layer)
                            #     with open("results/linalg_unflatten_layer_distance.txt", 'a') as file:
                            #         file.write(f"epoch_num： {epoch_num}, client_id: {client_id}, layer_id: {match.string}, layer_distance: {linalg_unflatten_layer_distance}\n")
                                
                                learnable_layers_distance+=layer_distance
                distance.append({client_id:learnable_layers_distance })

                # learnable_layers_distance = round((global_flat - client_flat).norm('fro').item(),4)
                # distance.append({client_id:learnable_layers_distance})

            # Sort client distance and according to selected percentage to select clients
            sorted_clients_distance = sorted(distance, key=lambda x:list(x.values())[0], reverse=True)

            # with open("results/sorted_model_layer_distance.txt", 'a') as file:
            #     for item in sorted_clients_distance:
            #         client_num,distance_value = list(item.items())[0]
            #         client_norm_info = "Global_round: {}, Client: {}, distance: {}\n".format(epoch_num,client_num, distance_value)
            #         file.write(client_norm_info)

            # Select the top 80% largest values along with their corresponding keys
            top_keys = [list(item.keys())[0] for item in sorted_clients_distance[:total_elements]]
            self._user_indices_overselected.extend(top_keys)

            with open("results/sorted_model_layer_distance.txt", 'a') as file:
                for item in sorted_clients_distance:
                    client_num,distance_value = list(item.items())[0]
                    if client_num in top_keys:
                        client_norm_info = "Global_round: {}, Client: {}, distance: {}\n".format(epoch_num,client_num, distance_value)
                        file.write(client_norm_info)

        print(f"client index: {self._user_indices_overselected}")
        return self._user_indices_overselected


@dataclass
class ActiveUserSelectorConfig:
    _target_: str = MISSING
    _recursive_: bool = False
    user_selector_seed: Optional[int] = None


@dataclass
class UniformlyRandomActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(UniformlyRandomActiveUserSelector)
    random_with_replacement: bool = False


@dataclass
class SequentialActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(SequentialActiveUserSelector)


@dataclass
class RandomRoundRobinActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(RandomRoundRobinActiveUserSelector)


@dataclass
class ImportanceSamplingActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(ImportanceSamplingActiveUserSelector)
    num_tries: int = 10


@dataclass
class RandomMultiStepActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(RandomMultiStepActiveUserSelector)
    random_with_replacement: bool = False
    gamma: int = 10
    milestones: List[int] = field(default_factory=list)


@dataclass
class LargestDistanceActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(LargestDistanceActiveUserSelector)

@dataclass
class ModelLayerActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(ModelLayerActiveUserSelector)


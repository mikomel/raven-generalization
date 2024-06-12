from abc import ABC, abstractmethod

import numpy as np
import torch


def create_raven_rule_encoder(name: str) -> "RavenRuleEncoder":
    if name == 'dense':
        return DenseRavenRuleEncoder()
    elif name == 'sparse':
        return SparseRavenRuleEncoder()
    else:
        raise ValueError(f"Can't create RavenRuleEncoder with name {name}. Choose one from: {{dense, sparse}}")


class RavenRuleEncoder(ABC):
    def __init__(self, num_components: int = 2, num_rules: int = 4, num_attributes: int = 5):
        self.num_components = num_components
        self.num_rules = num_rules
        self.num_attributes = num_attributes

    @staticmethod
    @abstractmethod
    def encode(data: np.array) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def encoding_size() -> int:
        pass


class DenseRavenRuleEncoder(RavenRuleEncoder):
    @staticmethod
    def encode(data: np.array) -> torch.Tensor:
        return torch.from_numpy(data['meta_target']).float()

    @staticmethod
    def encoding_size() -> int:
        return 9


class SparseRavenRuleEncoder(RavenRuleEncoder):
    """
    meta_matrix:
      row format: ["Constant", "Progression", "Arithmetic", "Distribute_Three", "Number", "Position", "Type", "Size", "Color"]
      rows[0:4] -- first component
      rows[4:8] -- second component
      rows[8:12] -- third component (Mesh)
    """

    def encode(self, data: np.array) -> torch.Tensor:
        meta_matrix = data['meta_matrix']
        rules = torch.zeros(self.encoding_size())
        for component in range(self.num_components):
            for row in meta_matrix[component * 4:(component + 1) * 4]:
                rule = row[:self.num_rules].argmax()
                attributes = np.where(row[self.num_rules:] == 1)[0]
                indices = component * (self.num_rules * self.num_attributes) + rule * self.num_attributes + attributes
                rules[indices] = 1
        return rules

    def encoding_size(self) -> int:
        return self.num_components * self.num_rules * self.num_attributes


class RavenRuleDecoder(RavenRuleEncoder):
    RULES = ['Constant', 'Progression', 'Arithmetic', 'DistributeThree']
    ATTRIBUTES = ['Number', 'Position', 'Type', 'Size', 'Color']

    def __init__(self, verbose: bool = True, num_components: int = 2, num_rules: int = 4, num_attributes: int = 5):
        super().__init__(num_components, num_rules, num_attributes)
        self.verbose = verbose

    def encode(self, data: np.array) -> torch.Tensor:
        meta_matrix = data['meta_matrix']
        for component in range(self.num_components):
            for row in meta_matrix[component * 4:(component + 1) * 4]:
                rule = row[:self.num_rules].argmax()
                attributes = np.where(row[self.num_rules:] == 1)[0]
                if len(attributes) > 0:
                    for attribute in attributes:
                        if self.verbose:
                            print(f"Component {component} - {self.RULES[rule]} {self.ATTRIBUTES[attribute]}")
        return meta_matrix

    def encoding_size(self) -> int:
        return -1

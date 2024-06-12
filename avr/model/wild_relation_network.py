import torch
from torch import nn

from avr.model.neural_blocks import ConvBnRelu, Identity, GroupObjectsIntoTriples, GroupObjectsIntoPairs, \
    DeepLinearBNReLU, Sum


class WildRelationNetwork(nn.Module):
    def __init__(
            self,
            num_channels: int = 32,
            embedding_size: int = 128,
            image_size: int = 160,
            use_object_triples: bool = False,
            use_layer_norm: bool = False,
            g_depth: int = 3,
            f_depth: int = 2,
            f_dropout_probability: float = 0.0):
        super(WildRelationNetwork, self).__init__()
        self.embedding_size = embedding_size
        if use_object_triples:
            self.group_objects = GroupObjectsIntoTriples()
            self.group_objects_with = GroupObjectsIntoTriplesWith()
        else:
            self.group_objects = GroupObjectsIntoPairs()
            self.group_objects_with = GroupObjectsIntoPairsWith()

        self.cnn = nn.Sequential(
            ConvBnRelu(1, num_channels, kernel_size=3, stride=2, padding=1),
            ConvBnRelu(num_channels, num_channels, kernel_size=3, padding=1),
            ConvBnRelu(num_channels, num_channels, kernel_size=3, padding=1),
            ConvBnRelu(num_channels, num_channels, kernel_size=3, padding=1),
            nn.Flatten(start_dim=-3, end_dim=-1))
        conv_dimension = num_channels * ((40 * (image_size // 80)) ** 2)
        self.object_tuple_size = (3 if use_object_triples else 2) * (conv_dimension + 9)
        self.tag_panel_embeddings = TagPanelEmbeddings()
        self.g = nn.Sequential(
            DeepLinearBNReLU(g_depth, self.object_tuple_size, embedding_size, change_dim_first=True),
            Sum(dim=1))
        self.norm = nn.LayerNorm(embedding_size) if use_layer_norm else Identity()
        self.f = nn.Sequential(
            DeepLinearBNReLU(f_depth, embedding_size, embedding_size),
            nn.Dropout(f_dropout_probability),
            nn.Linear(embedding_size, embedding_size))

    def forward(self, context: torch.Tensor, answers: torch.Tensor, **kwargs) -> torch.Tensor:
        num_context_panels = context.size(1)
        num_answer_panels = answers.size(1)
        x = torch.cat([context, answers], dim=1)
        batch_size, num_panels, num_channels, height, width = x.shape
        x = x.view(batch_size * num_panels, num_channels, height, width)
        x = self.cnn(x)
        x = x.view(batch_size, num_panels, -1)
        x = self.tag_panel_embeddings(x, num_context_panels)
        context_objects = x[:, :num_context_panels, :]
        choice_objects = x[:, num_context_panels:, :]
        context_pairs = self.group_objects(context_objects)
        context_g_out = self.g(context_pairs)
        f_out = torch.zeros((batch_size, num_answer_panels, self.embedding_size), device=x.device).type_as(x)
        for i in range(num_answer_panels):
            context_choice_pairs = self.group_objects_with(context_objects, choice_objects[:, i, :])
            context_choice_g_out = self.g(context_choice_pairs)
            relations = context_g_out + context_choice_g_out
            relations = self.norm(relations)
            f_out[:, i, :] = self.f(relations)
        return f_out


class GroupObjectsIntoPairsWith(nn.Module):
    def forward(self, objects: torch.Tensor, object: torch.Tensor) -> torch.Tensor:
        _, num_objects, _ = objects.size()
        return torch.cat([
            objects,
            object.unsqueeze(1).repeat(1, num_objects, 1)
        ], dim=2)


class GroupObjectsIntoTriplesWith(nn.Module):
    def forward(self, objects: torch.Tensor, object: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects, object_size = objects.size()
        return torch.cat([
            objects.unsqueeze(1).repeat(1, num_objects, 1, 1),
            objects.unsqueeze(2).repeat(1, 1, num_objects, 1),
            object.unsqueeze(1).unsqueeze(2).repeat(1, num_objects, num_objects, 1)
        ], dim=3).view(batch_size, num_objects ** 2, 3 * object_size)


class TagPanelEmbeddings(nn.Module):
    """ Tags panel embeddings with their absolute coordinates. """

    def forward(self, panel_embeddings: torch.Tensor, num_context_panels: int) -> torch.Tensor:
        """
        Concatenates a one-hot encoded vector to each panel.
        The concatenated vector indicates panel absolute position in the RPM.
        :param panel_embeddings: a tensor of shape (batch_size, num_panels, embedding_size)
        :return: a tensor of shape (batch_size, num_panels, embedding_size + 9)
        """
        batch_size, num_panels, _ = panel_embeddings.shape
        tags = torch.zeros((num_panels, 9), device=panel_embeddings.device).type_as(panel_embeddings)
        tags[:num_context_panels, :num_context_panels] = torch.eye(num_context_panels, device=panel_embeddings.device).type_as(panel_embeddings)
        if num_panels > num_context_panels:
            tags[num_context_panels:, num_context_panels] = torch.ones(num_panels - num_context_panels, device=panel_embeddings.device).type_as(panel_embeddings)
        tags = tags.expand((batch_size, -1, -1))
        return torch.cat([panel_embeddings, tags], dim=2)

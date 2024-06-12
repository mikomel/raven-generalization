import torch
from torch import nn

from avr.model.neural_blocks import ConvBnRelu, Identity


class CnnLstm(nn.Module):
    def __init__(
            self,
            num_channels: int = 32,
            embedding_size: int = 128,
            image_size: int = 160,
            use_layer_norm: bool = False):
        super(CnnLstm, self).__init__()
        self.embedding_size = embedding_size
        conv_dimension = num_channels * ((40 * (image_size // 80)) ** 2)
        self.cnn = nn.Sequential(
            ConvBnRelu(1, num_channels, kernel_size=3, stride=2, padding=1),
            ConvBnRelu(num_channels, num_channels, kernel_size=3, padding=1),
            ConvBnRelu(num_channels, num_channels, kernel_size=3, padding=1),
            ConvBnRelu(num_channels, num_channels, kernel_size=3, padding=1),
            nn.Flatten(start_dim=-3, end_dim=-1))
        self.tag_panel_embeddings = TagPanelEmbeddings()
        self.lstm = nn.Sequential(
            nn.Linear(conv_dimension + 9, embedding_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embedding_size) if use_layer_norm else Identity(),
            nn.LSTM(batch_first=True, input_size=embedding_size, hidden_size=embedding_size, num_layers=1))
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(inplace=True),
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

        x = torch.cat([
            x[:, :num_context_panels, :].unsqueeze(dim=1).repeat(1, num_answer_panels, 1, 1),
            x[:, num_context_panels:, :].unsqueeze(dim=2)
        ], dim=2)
        x = x.view((batch_size * num_answer_panels), (num_context_panels + 1), -1)

        x, _ = self.lstm(x)
        x = x[:, -1, :].view(batch_size, num_answer_panels, -1)
        return x


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
        tags[:num_context_panels,
        :num_context_panels] = torch.eye(num_context_panels, device=panel_embeddings.device).type_as(panel_embeddings)
        tags[num_context_panels:,
        num_context_panels] = torch.ones(num_panels - num_context_panels, device=panel_embeddings.device).type_as(panel_embeddings)
        tags = tags.expand((batch_size, -1, -1))
        return torch.cat([panel_embeddings, tags], dim=2)

import pytest
import torch

from avr.model.cnn_lstm import CnnLstm
from avr.model.context_blind import (
    ContextBlindWildResNet,
)
from avr.model.copinet import CoPINet
from avr.model.cpcnet import CPCNet
from avr.model.pong import Pong
from avr.model.predrnet import PredRNet
from avr.model.relbase import RelBase
from avr.model.scl import SCL
from avr.model.sran import SRAN
from avr.model.stsn import STSN
from avr.model.wild_relation_network import WildRelationNetwork

BATCH_SIZE = 4
NUM_CONTEXT_PANELS = 8
NUM_ANSWER_PANELS = 5
NUM_CHANNELS = 1
IMAGE_SIZE = 80
EMBEDDING_SIZE = 128


@pytest.fixture
def context():
    return torch.rand(
        BATCH_SIZE, NUM_CONTEXT_PANELS, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE
    )


@pytest.fixture
def answers():
    return torch.rand(
        BATCH_SIZE, NUM_ANSWER_PANELS, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE
    )


def test_models(context: torch.tensor, answers: torch.tensor):
    for model in [
        ContextBlindWildResNet(
            num_answers=NUM_ANSWER_PANELS, resnet_name="resnet18"
        ),
        ContextBlindWildResNet(
            num_answers=NUM_ANSWER_PANELS, resnet_name="resnet50"
        ),
        CnnLstm(image_size=IMAGE_SIZE),
        CoPINet(),
        CPCNet(image_size=IMAGE_SIZE),
        PredRNet(num_classes=NUM_ANSWER_PANELS),
        RelBase(),
        SCL(image_size=IMAGE_SIZE),
        SRAN(),
        WildRelationNetwork(image_size=IMAGE_SIZE),
    ]:
        y = model(context, answers)
        assert y.shape == (BATCH_SIZE, NUM_ANSWER_PANELS, EMBEDDING_SIZE)


def test_stsn(context: torch.tensor, answers: torch.tensor):
    model = STSN(embedding_size=EMBEDDING_SIZE)
    combined_reconstruction, reconstruction, mask, slots, attn, y_hat = model(
        context, answers
    )
    assert combined_reconstruction.shape == (
        BATCH_SIZE,
        NUM_CONTEXT_PANELS + NUM_ANSWER_PANELS,
        NUM_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )
    assert reconstruction.shape == (
        BATCH_SIZE,
        NUM_CONTEXT_PANELS + NUM_ANSWER_PANELS,
        model.encoder.num_slots,
        NUM_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )
    assert mask.shape == (
        BATCH_SIZE,
        NUM_CONTEXT_PANELS + NUM_ANSWER_PANELS,
        model.encoder.num_slots,
        NUM_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )
    assert slots.shape == (
        BATCH_SIZE,
        NUM_CONTEXT_PANELS + NUM_ANSWER_PANELS,
        model.encoder.num_slots,
        model.encoder.feature_dim,
    )
    assert attn.shape == (
        BATCH_SIZE,
        NUM_CONTEXT_PANELS + NUM_ANSWER_PANELS,
        model.encoder.num_slots,
        NUM_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )
    assert y_hat.shape == (BATCH_SIZE, NUM_ANSWER_PANELS, EMBEDDING_SIZE)


def test_pong(context: torch.tensor, answers: torch.tensor):
    model = Pong()
    y = model(context, answers)
    assert y.shape == (BATCH_SIZE, NUM_ANSWER_PANELS, EMBEDDING_SIZE)


def test_pong_ablations(context: torch.tensor, answers: torch.tensor):
    ablations = [
        {
            "reasoner_use_row_group_conv": False,
            "reasoner_use_row_pair_group_conv": False,
        },
        {
            "reasoner_group_conv_use_norm": False,
        },
        {
            "reasoner_group_conv_use_pre_norm": True,
        },
    ]
    for hparams in ablations:
        model = Pong(**hparams)
        y = model(context, answers)
        assert y.shape == (BATCH_SIZE, NUM_ANSWER_PANELS, EMBEDDING_SIZE)

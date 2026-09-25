import numpy as np
import pytest
from tabletop.vlm import parse_response
from tabletop.policy import to_controller_actions

@pytest.mark.parametrize("text", [
    '{"type":"skill","skill":"pick","object":"yellow"}',
    '{"type":"skill","skill":"stack","object":"red","target":"red"}',
    '{"type":"skill","skill":"shell","object":"red"}',
    '{"type":"skill","skill":"pick","object":"red","code":"print(1)"}',
    '{"type":"skill","skill":"place","object":"red","target":"far"}',
    'not json', '[]', '{"type":"answer","text":3}',
])
def test_invalid_model_output(text):
    with pytest.raises(ValueError):
        parse_response(text)


def test_action_units():
    assert np.allclose(to_controller_actions([[0.025, 0, 0, 0, 0, 0.25, -1]]),
                       [[0.5, 0, 0, 0, 0, 0.5, -1]])

@pytest.mark.parametrize("chunk", [np.zeros((21, 7)), [[float('nan')] * 7], [[1] * 7]])
def test_invalid_action_chunk(chunk):
    with pytest.raises(ValueError):
        to_controller_actions(chunk)

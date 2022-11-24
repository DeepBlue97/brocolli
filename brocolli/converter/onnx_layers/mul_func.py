from loguru import logger
from onnx import helper
from torch.fx.node import Node
from onnx import TensorProto as tp
import numbers
import numpy as np

from brocolli.converter.onnx_layers.base_layer import BaseLayer


class MulFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(MulFunc, self).__init__(source_node, module, auto_gen)

    def generate_node(self, name=None, params=None, attr_dict=None):

        assert len(self._source_node.args) == 2
        if len(self._source_node.all_input_nodes) == 1:
            if isinstance(self._source_node.args[0], Node):
                assert isinstance(self._source_node.args[1], numbers.Number)
                self.generate_params(np.array([self._source_node.args[1]]))
            else:
                assert isinstance(self._source_node.args[0], numbers.Number)
                self.generate_params(np.array([self._source_node.args[0]]))

        node = helper.make_node("Mul", self._in_names, self._out_names, self._name)

        logger.info("mul_layer: " + self._name + " created")
        self._node.append(node)

    def generate_params(self, params):
        self.create_params(self._name + "_mul_constant", params, tp.FLOAT)

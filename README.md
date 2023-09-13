# brocolli

torch fx based pytorch model converter, including pytorch2caffe, pytorch2onnx.  
torch fx based pytorch model quantizier.

# Installation
```
pip install brocolli
```

# How to use
* torch2caffe
    * caffe installation
    ```bash
    pip install brocolli-caffe
    ```

    ```
    import torchvision.models as models
    from brocolli.converter.pytorch_caffe_parser import PytorchCaffeParser

    net = models.alexnet(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    pytorch_parser = PytorchCaffeParser(net, x)
    pytorch_parser.convert()
    pytorch_parser.save('alexnet')
    ```
    run this script until you see "accuracy test passed" on screen, then you can get alexnet.caffemodel and alexnet.prototxt under under current folder.

* torch2onnx
    ```
    import torchvision.models as models
    from brocolli.converter.pytorch_onnx_parser import PytorchOnnxParser

    net = models.alexnet(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    pytorch_parser = PytorchOnnxParser(net, x)
    pytorch_parser.convert()
    pytorch_parser.save('alexnet.onnx')
    ```
    run this script until you see "accuracy test passed" on screen, then you can get alexnet.onnx under current folder.

# Contact
 QQ Group: 597059928
 
 ![image](https://raw.githubusercontent.com/inisis/brocolli/master/imgs/QGRPOUP.png)
 
# Show your support
  Give a 🌟 if this project helpes~

# Bilibili 
## Symmetric Quantization 对称量化

将绝对值最大的值的绝对值的负值即$-max(|X_f|)$映射到-128，正值$max(|X_f|)$映射到127，故有一个半轴是没有充分利用的。

scale的计算公式：

$$scale = \frac{x}{2^{num\_bits}-1}\quad if \quad min(x)>0 \quad scale=\frac{x}{2^{num\_bits-1}-1}$$

缺点：低比特利用率。

优点：对于ReLU，比特利用率较高。


## Pytorch Quantization

float模型流程：

x => conv => output

int模型流程：

x => Qx => Qconv => DQoutput


输入量化：

$$ Qx = roundclip(\frac{x}{scale_x}) $$

卷积层量化：
$$ conv_{output} = roundclip(Qx * roundclip(\frac{w}{scale_{wt}})+roundclip(\frac{b}{scale_x*scale_{wt}})) $$

输出映射回float：
$$output = conv_{output} * scale_{out}$$


以上scale均为统计得到。


## torch fx

Problems:
1. eager mode, no directed acyclic graph(DAG).
2. lack of flexible api to modify model.

torch fx:
1. fx is a toolkit for developers to use to transform nn.Module
2. three components: symbolic(符号) tracer; intermediate representation(graph); python code generation(code)

### GraphModule

1. GraphModule inherets nn.Module
2. GraphModule has graph and code,(auto gen)
3. graph can be iterated to get node, and node correspond to raw module or function or method.
4. GraphModule has same modules and state_dict as raw models.
5. GraphModule can be deployed without model python file.



scale_act：输入的尺度

scale_wt：权重的尺度，conv的话是逐通道的

scale_out：输出的尺度

1. prepare qconfig for each node;
2. insert observer for each node, observer is configured by qconfig;
3. run calibration(activation quantization and weight&bias quantization);
4. convert float modules to quantized modules(Conv->QConv);
5. evaluate quantized model;
6. profile model;

## Tracer
```
from torch.fx import Tracer, Graph, Node

class BrocolliTracer(Tracer):
    def is_leaf_module():
        pass
```

自定义BrocolliTracer继承了torch.fx的Tracer，is_leaf_module用于判断是否为叶子节点，从而防止将一个大算子（如LayerNorm）拆分成几个小算子。

# The Ultimate Guide to Deep Learning Model Quantization and Quantization-Aware Training

https://deci.ai/quantization-and-quantization-aware-training/

## Calibration
Squeeze(estimating the range) is done with a process called calibration.

## scale quantization

- max: use the maximum of the absolute value distribution as the α.
- percentile: use the α that is corresponding to the k-th percentile of the absolute value distribution.
- entropy: calculate the KL divergence between the quantized distribution and the original one.

## Post-Training Quantization and Quantization-Aware Training

Pretrain => PTQ => QAT

## Types of Quantization: Naive, Hybrid, and Selective

- Naive: All operators are quantized to INT8 precision, and are calibrated using the same method.
- Hybrid: Some operators are quantized to INT8 precision, and some are left in mode representative data type like FP16 or FP32.
- **Selective☆**: Some operators are quantized to INT8 precision, with different calibration methods and different granularity(粒度) (per channel or per tensor), residuals are quantized to INT8, as well as sensitive and non-friendly layers remain in FP16 precision.


## Best Practices: How to Achieve FP32 Accuracy with INT8 Inference Speed

### Tips for Post-Training Quantization

- Use per-channel granularity for weights and per-tensor for activations
- Quantize residual connections separately by replacing blocks
- Identify sensitive layers and skip them from quantization

### Tips for Quantization-Aware Training
- Start with the best-performing calibrated PTQ model
- Fine-tune for around 10% of the original training schedule
- Use cosine annealing LR schedule starting at 1% of the initial training LR
- Use SGD optimizer with momentum instead of ADAM or RMSProp


### Cosine Annealing 余弦退火
$$
lr = lr_{min} + 0.5 * (lr_{max} - lr_{min}) * [1 + cos(\frac{epoch * pi}{T_{max}})]
$$


## Zero Hustle(零负担) Selective PTQ and QAT with SuperGradients on NVIDIA TensorRT 8.4+


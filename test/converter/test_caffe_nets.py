import os

os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors

import pytest
import warnings

import torchvision.models as models

from bin.converter.pytorch2caffe import Runner

FUSE = True

os.makedirs("tmp", exist_ok=True)


def test_alexnet(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    net = models.alexnet(pretrained=False)
    runner = Runner("alexnet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


def test_resnet18(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    net = models.resnet18(pretrained=False)
    runner = Runner("resnet18", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


def test_squeezenet(shape=(1, 3, 227, 227), opset_version=9, fuse=FUSE):
    net = models.squeezenet1_0(pretrained=False)
    runner = Runner("squeezenet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


def test_googlenet(shape=(1, 3, 224, 224), opset_version=13, fuse=FUSE):
    net = models.googlenet(pretrained=False)
    runner = Runner("googlenet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


def test_mobilenet_v2(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    net = models.mobilenet_v2(pretrained=False)
    runner = Runner("mobilenet", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


def test_mobilenet_v3(shape=(1, 3, 224, 224), opset_version=13, fuse=FUSE):
    net = models.mobilenet_v3_small(pretrained=False)
    runner = Runner("mobilenet_v3", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


def test_densenet121(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    net = models.densenet121(pretrained=False)
    runner = Runner("densenet121", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


def test_densenet161(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    net = models.densenet161(pretrained=False)
    runner = Runner("densenet161", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


def test_shufflenet(shape=(1, 3, 224, 224), opset_version=9, fuse=FUSE):
    net = models.shufflenet_v2_x1_0(pretrained=False)
    runner = Runner("shufflenet_v2_x1_0", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


def test_ssd300_vgg16(shape=(1, 3, 300, 300), opset_version=13, fuse=FUSE):
    from custom_models.ssd import build_ssd

    net = build_ssd("export")
    runner = Runner("ssd300_vgg16", net, shape, opset_version, fuse)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


def test_yolov5(shape=(1, 3, 640, 640), opset_version=13, fuse=FUSE):
    import torch
    concrete_args = {"augment": False, "profile": False, "visualize": False}
    net = torch.hub.load(
        "ultralytics/yolov5",
        "yolov5s",
        autoshape=False,
        pretrained=False,
        device=torch.device("cpu"),
    )

    class Identity(torch.nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            for i in range(self.nl):
                x[i] = self.m[i](x[i])
                bs, _, ny, nx = x[i].shape
                x[i] = (
                    x[i]
                    .view(bs, self.na, self.no, ny, nx)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )

            return x

    name, _ = list(net.model.named_children())[-1]
    identity = Identity()
    detect = getattr(net.model, name)
    identity.__dict__.update(detect.__dict__)
    setattr(net.model, name, identity)

    runner = Runner("yolov5", net, shape, opset_version, fuse, concrete_args)
    runner.pyotrch_inference()
    runner.convert()
    runner.caffe_inference()
    runner.check_result()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(["-p", "no:warnings", "-v", "test/converter/test_caffe_nets.py"])

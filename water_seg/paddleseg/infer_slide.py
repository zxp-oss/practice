# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections.abc
from itertools import combinations

import numpy as np
import cv2
import paddle
import paddle.nn.functional as F


def reverse_transform(pred, trans_info, mode='nearest'):
    """recover pred to origin shape"""
    intTypeList = [paddle.int8, paddle.int16, paddle.int32, paddle.int64]
    dtype = pred.dtype
    for item in trans_info[::-1]:
        if isinstance(item[0], list):
            trans_mode = item[0][0]
        else:
            trans_mode = item[0]
        if trans_mode == 'resize':
            h, w = item[1][0], item[1][1]
            if paddle.get_device() == 'cpu' and dtype in intTypeList:
                pred = paddle.cast(pred, 'float32')
                pred = F.interpolate(pred, [h, w], mode=mode)
                pred = paddle.cast(pred, dtype)
            else:
                pred = F.interpolate(pred, [h, w], mode=mode)
        elif trans_mode == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred


def flip_combination(flip_horizontal=False, flip_vertical=False):
    """
    Get flip combination.

    Args:
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.

    Returns:
        list: List of tuple. The first element of tuple is whether to flip horizontally,
            and the second is whether to flip vertically.
    """

    flip_comb = [(False, False)]
    if flip_horizontal:
        flip_comb.append((True, False))
    if flip_vertical:
        flip_comb.append((False, True))
        if flip_horizontal:
            flip_comb.append((True, True))
    return flip_comb


def tensor_flip(x, flip):
    """Flip tensor according directions"""
    if flip[0]:
        x = x[:, :, :, ::-1]
    if flip[1]:
        x = x[:, :, ::-1, :]
    return x


def slide_inference(model, im, crop_size, stride):
    """
    Infer by sliding window.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        crop_size (tuple|list). The size of sliding window, (w, h).
        stride (tuple|list). The size of stride, (w, h).

    Return:
        Tensor: The logit of input image.
    """
    h_im, w_im = im.shape[-2:]
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride
    # calculate the crop nums
    rows = int(np.ceil(1.0 * (h_im - h_crop) / h_stride)) + 1
    cols = int(np.ceil(1.0 * (w_im - w_crop) / w_stride)) + 1
    # prevent negative sliding rounds when imgs after scaling << crop_size
    rows = 1 if h_im <= h_crop else rows
    cols = 1 if w_im <= w_crop else cols
    # TODO 'Tensor' object does not support item assignment. If support, use tensor to calculation.
    final_logit = None
    count = np.zeros([1, 1, h_im, w_im])
    for r in range(rows):
        for c in range(cols):
            h1 = r * h_stride
            w1 = c * w_stride
            h2 = min(h1 + h_crop, h_im)
            w2 = min(w1 + w_crop, w_im)
            h1 = max(h2 - h_crop, 0)
            w1 = max(w2 - w_crop, 0)
            im_crop = im[:, :, h1:h2, w1:w2]
            logits = model(im_crop)
            if not isinstance(logits, collections.abc.Sequence):
                raise TypeError(
                    "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                    .format(type(logits)))
            logit = logits[0].numpy()
            if final_logit is None:
                final_logit = np.zeros([1, logit.shape[1], h_im, w_im])
            final_logit[:, :, h1:h2, w1:w2] += logit[:, :, :h2 - h1, :w2 - w1]
            count[:, :, h1:h2, w1:w2] += 1
    if np.sum(count == 0) != 0:
        raise RuntimeError(
            'There are pixel not predicted. It is possible that stride is greater than crop_size'
        )
    final_logit = final_logit / count
    final_logit = paddle.to_tensor(final_logit)
    return final_logit

import argparse
import os

import paddle

from paddleseg.core import predict
from paddleseg.cvlibs import Config, SegBuilder, manager
from paddleseg.transforms import Compose
from paddleseg.utils import get_image_list, get_sys_env, logger, utils


def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # Common params
    parser.add_argument("--config",default="/home/zxp/test_project/PaddleSeg-release-2.9/my_configs/my_paddleseg_segnext_five_water.yaml" ,help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        default="/home/zxp/test_project/PaddleSeg-release-2.9/output_segnext_five_water/best_model/model.pdparams",
        help='The path of trained weights for prediction.',
        type=str)
    parser.add_argument(
        '--image_path',
        default="images/GF2_PMS1__L1A0000564539-MSS1_tile_0.jpg",
        help='The image to predict, which can be a path of image, or a file list containing image paths, or a directory including images',
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the predicted results.',
        type=str,
        default='./output/result')
    parser.add_argument(
        '--device',
        help='Set the device place for predicting model.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)
    parser.add_argument(
        '--device_id',
        help='Set the device id for predicting model.',
        default=0,
        type=int)

    # Data augment params
    parser.add_argument(
        '--aug_pred',
        help='Whether to use mulit-scales and flip augment for prediction',
        action='store_true')
    parser.add_argument(
        '--scales',
        nargs='+',
        help='Scales for augment, e.g., `--scales 0.75 1.0 1.25`.',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # Sliding window evaluation params
    parser.add_argument(
        '--is_slide',
        help='Whether to predict images in sliding window method',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        nargs=2,
        help='The crop size of sliding window, the first is width and the second is height.'
        'For example, `--crop_size 512 512`',
        type=int)
    parser.add_argument(
        '--stride',
        nargs=2,
        help='The stride of sliding window, the first is width and the second is height.'
        'For example, `--stride 512 512`',
        type=int)

    # Custom color map
    parser.add_argument(
        '--custom_color',
        nargs='+',
        help='Save images with a custom color map. Default: None, use paddleseg\'s default color map.',
        type=int)

    # Set multi-label mode
    parser.add_argument(
        '--use_multilabel',
        action='store_true',
        default=False,
        help='Whether to enable multilabel mode. Default: False.')

    return parser.parse_args()


def merge_test_config(cfg, args):
    test_config = cfg.test_config
    if 'aug_eval' in test_config:
        test_config.pop('aug_eval')
    if 'auc_roc' in test_config:
        test_config.pop('auc_roc')
    if args.aug_pred:
        test_config['aug_pred'] = args.aug_pred
        test_config['scales'] = args.scales
        test_config['flip_horizontal'] = args.flip_horizontal
        test_config['flip_vertical'] = args.flip_vertical
    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride
    if args.custom_color:
        test_config['custom_color'] = args.custom_color
    if args.use_multilabel:
        test_config['use_multilabel'] = args.use_multilabel
    return test_config


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.config)
    builder = SegBuilder(cfg)
    test_config = merge_test_config(cfg, args)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    if args.device != 'cpu':
        device = f"{args.device}:{args.device_id}"
    else:
        device = args.device
    utils.set_device(device)

    model = builder.model
    transforms = Compose(builder.val_transforms)
    image_list, image_dir = get_image_list(args.image_path)
    logger.info('The number of images: {}'.format(len(image_list)))

    utils.utils.load_entire_model(model, args.model_path)
    model.eval()

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
        **test_config)
    slide_inference(
        model,
        im=

    )


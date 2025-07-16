# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class RepeatedAugmentationTransform:
    """Applies a transform multiple times.

    Input: {"input": <image>, ...}
    Output: {"input0": <augmented tensor>, "input1": <augmented tensor>, ...}
    """

    def __init__(self, transform, copies=2):
        self.transform = transform
        self.copies = copies

    def __call__(self, img):

        imgs = []
        for i in range(self.copies):
            imgs.append(self.transform(img))
        
        return imgs

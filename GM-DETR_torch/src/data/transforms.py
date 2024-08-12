""""by lyuwenyu
"""


import torch 
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import datapoints

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from torchvision.transforms.v2.utils import is_simple_tensor, query_chw
from torchvision.transforms.v2._color import ColorJitter

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from PIL import Image 
from typing import Any, Dict, List, Optional

from src.core import register, GLOBAL_CONFIG

from torchvision.transforms.v2._transform import _RandomApplyTransform
from torchvision import datapoints, transforms as _transforms

from typing import Any, Callable, cast, Dict, List, Mapping, Optional, Sequence, Type, Union
import collections
from contextlib import suppress

__all__ = ['Compose', ]


RandomPhotometricDistort = register(T.RandomPhotometricDistort)
RandomZoomOut = register(T.RandomZoomOut)
# RandomIoUCrop = register(T.RandomIoUCrop)
RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)
ToImageTensor = register(T.ToImageTensor)
ConvertDtype = register(T.ConvertDtype)
SanitizeBoundingBox = register(T.SanitizeBoundingBox)
RandomCrop = register(T.RandomCrop)
Normalize = register(T.Normalize)


@register
class SanitizeBoundingBoxX(T.SanitizeBoundingBox):
    """[BETA] Remove degenerate/invalid bounding boxes and their corresponding labels and masks.

    .. v2betastatus:: SanitizeBoundingBox transform

    This transform removes bounding boxes and their associated labels/masks that:

    - are below a given ``min_size``: by default this also removes degenerate boxes that have e.g. X2 <= X1.
    - have any coordinate outside of their corresponding image. You may want to
      call :class:`~torchvision.transforms.v2.ClampBoundingBox` first to avoid undesired removals.

    It is recommended to call it at the end of a pipeline, before passing the
    input to the models. It is critical to call this transform if
    :class:`~torchvision.transforms.v2.RandomIoUCrop` was called.
    If you want to be extra careful, you may call it after all transforms that
    may modify bounding boxes but once at the end should be enough in most
    cases.

    Args:
        min_size (float, optional) The size below which bounding boxes are removed. Default is 1.
        labels_getter (callable or str or None, optional): indicates how to identify the labels in the input.
            It can be a str in which case the input is expected to be a dict, and ``labels_getter`` then specifies
            the key whose value corresponds to the labels. It can also be a callable that takes the same input
            as the transform, and returns the labels.
            By default, this will try to find a "labels" key in the input, if
            the input is a dict or it is a tuple whose second element is a dict.
            This heuristic should work well with a lot of datasets, including the built-in torchvision datasets.
    """


    @staticmethod
    def _get_dict_or_third_tuple_entry(inputs: Any) -> Mapping[str, Any]:
        # datasets outputs may be plain dicts like {"img": ..., "labels": ..., "bbox": ...}
        # or tuples like (img, {"labels":..., "bbox": ...})
        # This hacky helper accounts for both structures.
        if isinstance(inputs, tuple):
            inputs = inputs[2]                  # change 1 to 2, to get

        if not isinstance(inputs, collections.abc.Mapping):
            raise ValueError(
                f"If labels_getter is a str or 'default', "
                f"then the input to forward() must be a dict or a tuple whose second element is a dict."
                f" Got {type(inputs)} instead."
            )
        return inputs

    @staticmethod
    def _find_labels_default_heuristic(inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        # Tries to find a "labels" key, otherwise tries for the first key that contains "label" - case insensitive
        # Returns None if nothing is found
        inputs = SanitizeBoundingBoxX._get_dict_or_third_tuple_entry(inputs)
        candidate_key = None
        with suppress(StopIteration):
            candidate_key = next(key for key in inputs.keys() if key.lower() == "labels")
        if candidate_key is None:
            with suppress(StopIteration):
                candidate_key = next(key for key in inputs.keys() if "label" in key.lower())
        if candidate_key is None:
            raise ValueError(
                "Could not infer where the labels are in the sample. Try passing a callable as the labels_getter parameter?"
                "If there are no samples and it is by design, pass labels_getter=None."
            )
        return inputs[candidate_key]
    

@register
class Compose(T.Compose):
    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)(**op)
                    transforms.append(transfom)
                    # op['type'] = name
                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]
 
        super().__init__(transforms=transforms)


@register
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):
    _transformed_types = (
        Image.Image,
        datapoints.Image,
        datapoints.Video,
        datapoints.Mask,
        datapoints.BoundingBox,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sz = F.get_spatial_size(flat_inputs[0])
        h, w = self.spatial_size[0] - sz[0], self.spatial_size[1] - sz[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        
        self.spatial_size = spatial_size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register
class ConvertBox(T.Transform):
    _transformed_types = (
        datapoints.BoundingBox,
    )
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if self.out_fmt:
            spatial_size = inpt.spatial_size
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = datapoints.BoundingBox(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)
        
        if self.normalize:
            inpt = inpt / torch.tensor(inpt.spatial_size[::-1]).tile(2)[None]

        return inpt


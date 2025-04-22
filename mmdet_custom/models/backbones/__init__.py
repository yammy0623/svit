# Copyright (c) Shanghai AI Lab. All rights reserved.
from .beit_adapter import BEiTAdapter
from .uniperceiver_adapter import UniPerceiverAdapter
from .vit_adapter import ViTAdapter
from .tome_vit_adapter import ToMeViTAdapter
from .vit_baseline import ViTBaseline
from .selective_vit_adapter import SelectiveVisionTransformer
from .evovit_adapter import EvoViTAdapter
from .demo_evovit_adapter import DemoEvoViTAdapter

__all__ = ['UniPerceiverAdapter', 'ViTAdapter', 'ToMeViTAdapter', 'ViTBaseline', 'BEiTAdapter',
           'SelectiveVisionTransformer', 'EvoViTAdapter', 'DemoEvoViTAdapter']

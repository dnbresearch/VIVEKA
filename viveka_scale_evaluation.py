"""
Compatibility wrapper: maps viveka_scale_evaluation imports to viveka.scale_evaluation.
"""
import sys, os

viveka_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, viveka_root)

from viveka.scale_evaluation import *

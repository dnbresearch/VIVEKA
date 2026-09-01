"""
Compatibility wrapper: maps community_hypothesis_v3 imports to viveka.community_detection.
Place this file alongside scripts that import community_hypothesis_v3.
"""
import sys, os

# Add the viveka package to path
viveka_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, viveka_root)

from viveka.community_detection import *

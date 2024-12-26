import os
from pathlib import Path

OPEN_EQA_UTILS_ROOT = os.path.dirname(os.path.abspath(__file__))
OPEN_EAQ_UTILS_CFG_DIR = os.path.join(OPEN_EQA_UTILS_ROOT, 'config')

os.environ['OPEN_EQA_UTILS_ROOT'] = OPEN_EQA_UTILS_ROOT
os.environ['OPEN_EAQ_UTILS_CFG_DIR'] = OPEN_EAQ_UTILS_CFG_DIR

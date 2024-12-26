import os
from pathlib import Path

UTILS_OPEN_EQA_ROOT = os.path.dirname(os.path.abspath(__file__))
UTILS_OPEN_EAQ_CFG_DIR = os.path.join(UTILS_OPEN_EQA_ROOT, 'config')

os.environ['UTILS_OPEN_EQA_ROOT'] = UTILS_OPEN_EQA_ROOT
os.environ['UTILS_OPEN_EAQ_CFG_DIR'] = UTILS_OPEN_EAQ_CFG_DIR

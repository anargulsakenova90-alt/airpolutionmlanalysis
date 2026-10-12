from pathlib import Path
import os
CODE = Path(__file__).resolve().parent
OUTPUT = Path(os.environ.get('ALMATY_OUTPUT', str(CODE / 'output'))).resolve()
RAW = Path(os.environ.get('ALMATY_RAW', str(OUTPUT / 'raw'))).resolve()
SOURCE = Path(os.environ.get('ALMATY_SOURCE', str(CODE / 'input'))).resolve()

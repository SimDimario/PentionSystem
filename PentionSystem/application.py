import os
import sys

from ui.state import init_state
from ui.components import ui_init


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


init_state()
ui_init()

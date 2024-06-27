import importlib
import sys
import os
# Add the root of your project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'lang-seg')))

# Now import the module dynamically
module_name = 'lang-seg.extract_lseg_features'
module = importlib.import_module(module_name)

# Access the function
extract_save_lseg_features = getattr(module, 'extract_save_lseg_features', None)

extract_save_lseg_features(['/home/fangj1/Code/go_vocation/data/scene_example/color/1539.jpg', '/home/fangj1/Code/go_vocation/data/scene_example/color/1555.jpg'], devide=2)
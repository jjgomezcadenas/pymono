# config.py

from pathlib import Path
import os
#mono_5_5_2x0_all_dark_sipm_6_6

data_dir = Path("/Users/jjgomezcadenas/data/monolith/")
mono_dark_6x6 = os.path.join(data_dir, "dark_walls_6x6_pde_04_ng_50k")
mono_light_3x3 =os.path.join(data_dir,"reflecting_walls_3x3_pde_04_ng_50k")
mono_light_6x6 =os.path.join(data_dir,"reflecting_walls_6x6_pde_04_ng_50k")
mono_light_all_6x6 =os.path.join(data_dir,"reflecting_all_walls_6x6_pde_04_ng_25k")


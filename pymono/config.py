# config.py

from pathlib import Path
import os

tutorial_dir = Path("/Users/jjgomezcadenas/Projects/pymono/notebooks/tutorials/data/")
imagenette_dir = os.path.join(tutorial_dir,"imagenette2-160")
data_dir = Path("/Users/jjgomezcadenas/data/monolith/")
test_dir = Path("/Users/jjgomezcadenas/Projects/pymono/tests/")
CsI_6x6_fullwrap_50k_0MHzDC_PTFE_LUT_NX = os.path.join(data_dir, 
                                          "CsI_6x6_fullwrap_50k_0MHzDC_PTFE_LUT_NX")
CsI_6x6_sidewrap_50k_2MHzDC_PTFE_LUT_fano_NX = os.path.join(data_dir, 
                                          "CsI_6x6_sidewrap_50k_2MHzDC_PTFE_LUT_fano_NX")
CsITl_6x6_sidewrap_25k_2MHzDC_PTFE_LUT_NX = os.path.join(data_dir, 
                                          "CsITl_6x6_sidewrap_25k_2MHzDC_PTFE_LUT_NX")
BGO_6x6_fullwrap_4d5k_2MHzDC_PTFE_LUT_NX = os.path.join(data_dir, 
                                          "BGO_6x6_fullwrap_4.5k_2MHzDC_PTFE_LUT_NX")
BGO_6x6_fullwrap_4d5k_2MHzDC_PTFE_LUT_fano_NX = os.path.join(data_dir, 
                                          "BGO_6x6_fullwrap_4.5k_2MHzDC_PTFE_LUT_fano_NX")
mono_dark_6x6 = os.path.join(data_dir, "dark_walls_6x6_pde_04_ng_50k")
mono_light_3x3 =os.path.join(data_dir,"reflecting_walls_3x3_pde_04_ng_50k")
mono_light_6x6 =os.path.join(data_dir,"reflecting_walls_6x6_pde_04_ng_50k")
mono_light_all_6x6 =os.path.join(data_dir,"reflecting_all_walls_6x6_pde_04_ng_25k")
esr_light_6x6 = os.path.join(data_dir,"reflecting_walls_esr_6x6_pde_04_ng_50k")
n4_light_6x6 = os.path.join(data_dir,"reflecting_walls_6x6_pde_04_ng_50k_n4")


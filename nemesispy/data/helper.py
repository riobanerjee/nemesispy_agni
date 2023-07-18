# -*- coding: utf-8 -*-
import os
import numpy as np

### Reliably find the path to this folder
__location__ = os.path.realpath(
    os.path.dirname(__file__)
    )

### k-tables
# All k-tables are assummed to share the same g ordinates and gordinate
# quadrature weights below, unless stated otherwise

# g ordinates
#g_ordinates = np.array(
#    [0.0034357 , 0.01801404, 0.04388279, 0.08044151, 0.12683405,
#       0.18197316, 0.2445665 , 0.31314695, 0.3861071 , 0.46173674,
#       0.53826326, 0.6138929 , 0.68685305, 0.7554335 , 0.81802684,
#       0.87316597, 0.91955847, 0.9561172 , 0.981986  , 0.9965643 ],
#      dtype=np.float32)
# g ordinates quadrature weights
#del_g = np.array(
#    [0.008807  , 0.02030071, 0.03133602, 0.04163837, 0.05096506,
#       0.05909727, 0.06584432, 0.07104805, 0.0745865 , 0.07637669,
#       0.07637669, 0.0745865 , 0.07104805, 0.06584432, 0.05909727,
#       0.05096506, 0.04163837, 0.03133602, 0.02030071, 0.008807  ],
#      dtype=np.float32)

### Low resolution k-table, HST/WFC3 + Spizter wavelengths
ktable_path = os.path.join(__location__, "ktables")
lowres_file_paths = [
     'H2O_test.kta']
for ipath,path in enumerate(lowres_file_paths):
    lowres_file_paths[ipath] = os.path.join(ktable_path,path)

### CIA tables
cia_folder_path = os.path.join(__location__ , "cia")
cia_file_path = os.path.join(cia_folder_path,'exocia_hitran12_200-3800K.tab')

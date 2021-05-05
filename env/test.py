from pyquaternion import Quaternion
import numpy as np
R_dcm=np.array([[0.2057   ,-0.0057   ,-0.9786],
                [0.9784   ,-0.0222   ,0.2057],
               [-0.0229   ,-0.9997    ,0.0010]])
R_dcm=R_dcm.T      
R_quat= list(Quaternion(matrix=R_dcm))
R_quat= np.array(R_quat)
#R_quat=[0.5442   0.5538   0.4391  -0.4521]
print(R_quat)

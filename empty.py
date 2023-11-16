import nibabel as nib
import matplotlib.pyplot as plt

nii = nib.load('data/ADNI 2/002_S_0413/MPR____N3__Scaled/2006-05-19_16_17_47.0/I40657/ADNI_002_S_0413_MR_MPR____N3__Scaled_Br_20070216232854688_S14782_I40657.nii').get_fdata()
a = 1
print(nii.shape)

plt.figure()
plt.imshow(nii[:][:][100],cmap='gray')
plt.show()


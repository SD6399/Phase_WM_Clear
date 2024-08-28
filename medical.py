import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
#==================================
# load image (4D) [X,Y,Z_slice,time]

t1 = nib.load('D:\Task01_BrainTumour\imagesTr/BRATS_001.nii.gz')
data = t1.get_fdata()
print(data.shape)
plt.imshow(data[:, :, data.shape[2] // 2,0].T, cmap='Greys_r')


plt.show()
"""
nii_data = nii_img.get_fdata()

fig, ax = plt.subplots(number_of_frames, number_of_slices, constrained_layout=True)
fig.canvas.set_window_title('4D Nifti Image')
fig.suptitle('4D_Nifti 10 slices 30 time Frames', fontsize=16)
# -------------------------------------------------------------------------------
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

for slice in range(number_of_slices):
    # if your data in 4D, otherwise remove this loop
    for frame in range(number_of_frames):
        ax[frame, slice].imshow(nii_data[:, :, slice, frame], cmap='gray', interpolation=None)
        ax[frame, slice].set_title("layer {} / frame {}".format(slice, frame))
        ax[frame, slice].axis('off')

plt.show()
"""
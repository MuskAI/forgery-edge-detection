"""
created by haoran
time:2020-7-15
input a mask to generate double edge
mask is an 0 1 mask
"""
import numpy as np
import skimage.morphology as dilation
def mask_to_outeedge(orignal_mask):
    """
    :param orignal_mask:
    :return: 01 outer_edge
    """
    print('We are in mask_to_outeedge function:')
    try:
        mask = orignal_mask
        print('the shape of mask is :', mask.shape)
        selem = np.ones((3,3))
        dst_8 = dilation.binary_dilation(mask, selem=selem)
        dst_8 = np.where(dst_8 == True,1,0)
        differece_8 = dst_8 - orignal_mask

    except Exception as e:
        print(e)

    return differece_8
if __name__ == '__main__':
    """
    the follow three lines code is used to check result，
    In practice ,won't use
    """
    test_mask = np.random.randint(0,100,(10,10))
    test_mask = np.where(test_mask >80, 1, 0)
    mask_to_outeedge(test_mask)

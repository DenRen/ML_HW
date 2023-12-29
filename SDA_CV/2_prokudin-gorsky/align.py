import numpy as np
from numpy.fft import fft2, ifft2, ifftshift


def calc_coord(left_img, right_img):
    assert left_img.shape == right_img.shape
    shape = left_img.shape
    
    left_fft = fft2(left_img, shape)
    right_fft = fft2(ifftshift(right_img), shape)
    conv = ifft2(left_fft * np.conj(right_fft), shape).real
    
    idx = np.argmax(conv)
    y, x = idx // left_img.shape[1], idx % left_img.shape[1]
    return np.array([y, x])

def align(img, g_coord):
    H, W = img.shape[0] // 3, img.shape[1]
    b_img, g_img, r_img = img[:H], img[H:2*H], img[2*H:3*H]
    
    def remove_border(img, part=0.1):
        size = int(part * img.shape[0])
        return img[size:-size, size:-size]
    
    r_img, g_img, b_img = map(remove_border, (r_img, g_img, b_img))
    
    def shift_rel_base(base_img, shift_img):
        center = np.array(base_img.shape) // 2
        coord = calc_coord(base_img, shift_img)
        shift = np.round(coord - center).astype(np.int32)
        return np.roll(shift_img, shift, axis=(0, 1)), shift

    r_img, r_shift = shift_rel_base(g_img, r_img)
    b_img, b_shift = shift_rel_base(g_img, b_img)

    max_y_shift, max_x_shift = np.max((np.abs(r_shift), np.abs(b_shift)), axis=0)
    
    rect = slice(max_y_shift, -max_y_shift), slice(max_x_shift, -max_x_shift)
    r_img, g_img, b_img = r_img[rect], g_img[rect], b_img[rect]
    
    aligned_img = np.array([r_img, g_img, b_img]).transpose(1, 2, 0)
    aligned_img /= aligned_img.max()

    b_row = g_coord[0] - b_shift[0] - H
    r_row = g_coord[0] - r_shift[0] + H
    b_col = g_coord[1] - b_shift[1]
    r_col = g_coord[1] - r_shift[1]
    
    return aligned_img, (b_row, b_col), (r_row, r_col)

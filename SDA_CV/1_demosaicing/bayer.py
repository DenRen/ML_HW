import numpy as np


def is_even_number(num) -> bool:
    return (num & 0b1) == False


def is_odd_number(num) -> bool:
    return not is_even_number(num)


def get_color_views(colored_img):
    return colored_img[..., 0], colored_img[..., 1], colored_img[..., 2]


def get_bayer_masks(n_rows, n_cols):  # Func for test
    green_mask = np.zeros([n_rows, n_cols], dtype=bool)
    if is_odd_number(n_cols):
        green_mask.reshape(green_mask.size)[::2] = True
    else:
        green_mask[::2, ::2] = True
        green_mask[1::2, 1::2] = True

    blue_mask, red_mask = np.zeros_like(green_mask), np.zeros_like(green_mask)
    blue_mask[1::2, ::2] = True
    red_mask[::2, 1::2] = True

    return np.dstack([red_mask, green_mask, blue_mask])


def get_colored_img_separately(raw_img, masks):
    r, g, b = get_color_views(masks)
    return raw_img * r, raw_img * g, raw_img * b


def get_T_stack_T(img_r, img_g, img_b):
    return np.array((img_r.T, img_g.T, img_b.T)).T


def get_colored_img(raw_img):  # Func for test
    masks = get_bayer_masks(*raw_img.shape)
    return get_T_stack_T(*get_colored_img_separately(raw_img, masks))


def bilinear_interpolation(colored_img):  # Func for test
    res = colored_img.astype(np.uint16)

    rect = colored_img[:-1, :-1]
    res[1:, :-1] += rect
    res[:-1, 1:] += rect

    rect = colored_img[1:, 1:]
    res[1:, :-1] += rect
    res[:-1, 1:] += rect

    rb = colored_img[..., 0::2]
    rb_res = res[..., 0::2]

    rb_res[1:, 1:] += rb[:-1, :-1]
    rb_res[:-1, :-1] += rb[1:, 1:]
    rb_res[1:, :-1] += rb[:-1, 1:]
    rb_res[:-1, 1:] += rb[1:, :-1]

    rb, g, b = get_color_views(res[1:-1, 1:-1])

    g[0::2, 1::2] >>= 2
    g[1::2, 0::2] >>= 2

    rb[0::2, 0::2] >>= 1
    rb[1::2, 1::2] >>= 1
    rb[0::2, 1::2] >>= 2

    b[0::2, 0::2] >>= 1
    b[1::2, 1::2] >>= 1
    b[1::2, 0::2] >>= 2

    return res.astype(np.uint8)


def normilize_color_range(clr_img, color_radius):
    clr_img += color_radius
    np.clip(clr_img, color_radius, 2 * color_radius, out=clr_img)
    clr_img -= color_radius


def improved_interpolation(raw_img):  # Func for test
    accum_dtype = np.uint16
    color_radius = 255 * 16

    masks = get_bayer_masks(*raw_img.shape)
    img_r, img_g, img_b = get_colored_img_separately(raw_img, masks)
    img_r = img_r.astype(accum_dtype)
    img_g = img_g.astype(accum_dtype)
    img_b = img_b.astype(accum_dtype)

    res_r, res_g, res_b = img_r.copy(), img_g.copy(), img_b.copy()

    # Process green
    g_inn0_slice_even = slice(0, None, 2), slice(0, None, 2)
    g_inn0_slice_odd = slice(1, None, 2), slice(1, None, 2)
    r_inn0_slice = slice(0, None, 2), slice(1, None, 2)
    b_inn0_slice = slice(1, None, 2), slice(0, None, 2)

    g_inn1_slice_even = slice(1, -1, 2), slice(1, -1, 2)
    g_inn1_slice_odd = slice(2, -1, 2), slice(2, -1, 2)
    r_inn1_slice = slice(2, -1, 2), slice(1, -1, 2)
    b_inn1_slice = slice(1, -1, 2), slice(2, -1, 2)

    g_inn2_slice_even = slice(2, -2, 2), slice(2, -2, 2)
    g_inn2_slice_odd = slice(3, -2, 2), slice(3, -2, 2)
    r_inn2_slice = slice(2, -2, 2), slice(3, -2, 2)
    b_inn2_slice = slice(3, -2, 2), slice(2, -2, 2)

    # 0.0 Mult knowen G by 2
    res_g[g_inn1_slice_even] <<= 1
    res_g[g_inn1_slice_odd] <<= 1

    # 1 G at R location
    # 1.1 Calc Red in R position
    res_g[r_inn2_slice] = img_r[r_inn2_slice]
    res_g[r_inn2_slice] <<= 2

    res_g[r_inn2_slice] -= img_r[:-4:2, 3:-2:2]     # on the 2 x Up    of r
    res_g[r_inn2_slice] -= img_r[4::2, 3:-2:2]      # on the 2 x Down  of r
    res_g[r_inn2_slice] -= img_r[2:-2:2, 1:-4:2]    # on the 2 x Left  of r
    res_g[r_inn2_slice] -= img_r[2:-2:2, 5::2]      # on the 2 x Right of r

    # 1.2 Adding Green in R position
    res_g[r_inn2_slice] += res_g[2:-2:2, 2:-3:2]  # on the Left  of r
    res_g[r_inn2_slice] += res_g[2:-2:2, 4:-1:2]  # on the Right of r
    res_g[r_inn2_slice] += res_g[1:-3:2, 3:-2:2]  # on the Up    of r
    res_g[r_inn2_slice] += res_g[3:-1:2, 3:-2:2]  # on the Down  of r

    # 2 G at B location
    # 2.1 Calc Blue in B position
    res_g[b_inn2_slice] = img_b[b_inn2_slice]
    res_g[b_inn2_slice] <<= 2

    res_g[b_inn2_slice] -= img_b[3:-2:2, :-4:2]   # on the 2 x Left  of b
    res_g[b_inn2_slice] -= img_b[3:-2:2, 4::2]    # on the 2 x Right of b
    res_g[b_inn2_slice] -= img_b[1:-4:2, 2:-2:2]  # on the 2 x Up    of b
    res_g[b_inn2_slice] -= img_b[5::2, 2:-2:2]    # on the 2 x Down  of b

    # 2.2 Calc Green in B position
    res_g[b_inn2_slice] += res_g[3:-2:2, 1:-3:2]  # on the Left  of b
    res_g[b_inn2_slice] += res_g[3:-2:2, 3:-1:2]  # on the Right of b
    res_g[b_inn2_slice] += res_g[2:-3:2, 2:-2:2]  # on the Up    of b
    res_g[b_inn2_slice] += res_g[4:-1:2, 2:-2:2]  # on the Down  of b

    normilize_color_range(res_g, color_radius)

    res_g[r_inn2_slice] >>= 3
    res_g[b_inn2_slice] >>= 3

    res_g[g_inn1_slice_even] >>= 1
    res_g[g_inn1_slice_odd] >>= 1

    np.clip(res_g, 0, 255, out=res_g)

    # 3 R at green in R row, B column and B row, R column
    # 3.1 Calc Green in G position
    res_r[g_inn2_slice_even] = img_g[g_inn2_slice_even]
    res_r[g_inn2_slice_even] *= 10
    res_r[g_inn2_slice_odd] = img_g[g_inn2_slice_odd]
    res_r[g_inn2_slice_odd] *= 10

    # Adding 1/2 G
    res_r[g_inn2_slice_even] += img_g[:-4:2, 2:-2:2]    # 2 x Up
    res_r[g_inn2_slice_even] += img_g[4::2, 2:-2:2]     # 2 x Down
    res_r[g_inn2_slice_odd] += img_g[3:-2:2, 1:-4:2]    # 2 x Left
    res_r[g_inn2_slice_odd] += img_g[3:-2:2, 5::2]      # 2 x Right

    # Substruct 1 G
    img_g[g_inn0_slice_even] <<= 1
    img_g[g_inn0_slice_odd] <<= 1

    res_r[g_inn2_slice_even] -= img_g[1:-3:2, 1:-3:2]  # Left  Up
    res_r[g_inn2_slice_even] -= img_g[1:-3:2, 3:-1:2]  # Right Up
    res_r[g_inn2_slice_even] -= img_g[3:-1:2, 1:-3:2]  # Left  Down
    res_r[g_inn2_slice_even] -= img_g[3:-1:2, 3:-1:2]  # Right Down
    res_r[g_inn2_slice_even] -= img_g[2:-2:2, :-4:2]   # 2 x Left
    res_r[g_inn2_slice_even] -= img_g[2:-2:2, 4::2]    # 2 x Right

    res_r[g_inn2_slice_odd] -= img_g[2:-3:2, 2:-3:2]  # Left  Up
    res_r[g_inn2_slice_odd] -= img_g[2:-3:2, 4:-1:2]  # Right Up
    res_r[g_inn2_slice_odd] -= img_g[4:-1:2, 2:-3:2]  # Left  Down
    res_r[g_inn2_slice_odd] -= img_g[4:-1:2, 4:-1:2]  # Right Down
    res_r[g_inn2_slice_odd] -= img_g[1:-4:2, 3:-2:2]  # 2 x Up
    res_r[g_inn2_slice_odd] -= img_g[5::2, 3:-2:2]    # 2 x Down

    # Don't forget!
    img_g[g_inn0_slice_even] >>= 1
    img_g[g_inn0_slice_odd] >>= 1

    # 3.2 Calc Red in G position
    img_r[r_inn1_slice] <<= 3

    res_r[g_inn2_slice_even] += img_r[2:-2:2, 1:-3:2]   # Left
    res_r[g_inn2_slice_even] += img_r[2:-2:2, 3:-1:2]   # Right
    res_r[g_inn2_slice_odd] += img_r[2:-3:2, 3:-2:2]    # Up
    res_r[g_inn2_slice_odd] += img_r[4:-1:2, 3:-2:2]    # Down

    # Don't forget! (-1 because exist next step)
    img_r[r_inn1_slice] >>= 3 - 2

    # 4 Calc Red in B position
    # 4.1 Add Blue
    res_r[b_inn2_slice] = img_b[b_inn2_slice]
    res_r[b_inn2_slice] *= 12

    img_b[b_inn0_slice] *= 3

    res_r[b_inn2_slice] -= img_b[1:-4:2, 2:-2:2]  # 2 x Up
    res_r[b_inn2_slice] -= img_b[5::2, 2:-2:2]    # 2 x Down
    res_r[b_inn2_slice] -= img_b[3:-2:2, :-4:2]   # 2 x Left
    res_r[b_inn2_slice] -= img_b[3:-2:2, 4::2]    # 2 x Right

    # Don't forget!
    img_b[b_inn0_slice] //= 3

    # 4.2 Substruct Red
    res_r[b_inn2_slice] += img_r[2:-3:2, 1:-3:2]  # Left  Up
    res_r[b_inn2_slice] += img_r[2:-3:2, 3:-1:2]  # Right Up
    res_r[b_inn2_slice] += img_r[4:-1:2, 1:-3:2]  # Left  Down
    res_r[b_inn2_slice] += img_r[4:-1:2, 3:-1:2]  # Right Down

    img_r[r_inn1_slice] >>= 2

    normilize_color_range(res_r, color_radius)

    res_r[g_inn2_slice_even] >>= 4
    res_r[g_inn2_slice_odd] >>= 4
    res_r[b_inn2_slice] >>= 4

    np.clip(res_r, 0, 255, out=res_r)

    # 5 Calc Blue in G pos ----------------------------------------

    # 5.1 Calc Green in G position
    res_b[g_inn2_slice_even] = img_g[g_inn2_slice_even]
    res_b[g_inn2_slice_even] *= 10
    res_b[g_inn2_slice_odd] = img_g[g_inn2_slice_odd]
    res_b[g_inn2_slice_odd] *= 10

    # Adding 1/2 G
    res_b[g_inn2_slice_even] += img_g[2:-2:2, :-4:2]  # 2 x Left
    res_b[g_inn2_slice_even] += img_g[2:-2:2, 4::2]   # 2 x Right
    res_b[g_inn2_slice_odd] += img_g[1:-4:2, 3:-2:2]  # 2 x Up
    res_b[g_inn2_slice_odd] += img_g[5::2, 3:-2:2]    # 2 x Down

    # Substruct 1 G
    img_g[g_inn0_slice_even] <<= 1
    img_g[g_inn0_slice_odd] <<= 1

    res_b[g_inn2_slice_even] -= img_g[1:-3:2, 1:-3:2]  # Left  Up
    res_b[g_inn2_slice_even] -= img_g[1:-3:2, 3:-1:2]  # Right Up
    res_b[g_inn2_slice_even] -= img_g[3:-1:2, 1:-3:2]  # Left  Down
    res_b[g_inn2_slice_even] -= img_g[3:-1:2, 3:-1:2]  # Right Down
    res_b[g_inn2_slice_even] -= img_g[:-4:2, 2:-2:2]   # 2 x Up
    res_b[g_inn2_slice_even] -= img_g[4::2, 2:-2:2]    # 2 x Down

    res_b[g_inn2_slice_odd] -= img_g[2:-3:2, 2:-3:2]  # Left  Up
    res_b[g_inn2_slice_odd] -= img_g[2:-3:2, 4:-1:2]  # Right Up
    res_b[g_inn2_slice_odd] -= img_g[4:-1:2, 2:-3:2]  # Left  Down
    res_b[g_inn2_slice_odd] -= img_g[4:-1:2, 4:-1:2]  # Right Down
    res_b[g_inn2_slice_odd] -= img_g[3:-2:2, 1:-4:2]  # 2 x Left
    res_b[g_inn2_slice_odd] -= img_g[3:-2:2, 5::2]    # 2 x Right

    # Don't forget!
    img_g[g_inn0_slice_even] >>= 1
    img_g[g_inn0_slice_odd] >>= 1

    # 5.2 Calc Red in G position
    img_b[b_inn1_slice] <<= 3

    res_b[g_inn2_slice_even] += img_b[1:-3:2, 2:-2:2]    # Up
    res_b[g_inn2_slice_even] += img_b[3:-1:2, 2:-2:2]    # Down
    res_b[g_inn2_slice_odd] += img_b[3:-2:2, 2:-3:2]   # Left
    res_b[g_inn2_slice_odd] += img_b[3:-2:2, 4:-1:2]   # Right

    # Don't forget! (-1 because exist next step)
    img_b[b_inn1_slice] >>= 3 - 2

    # 6 Calc Blue in R position
    # 6.1 Sub Red
    res_b[r_inn2_slice] = img_r[r_inn2_slice]
    res_b[r_inn2_slice] *= 12

    # 6.2 Add Red
    img_r[r_inn0_slice] *= 3

    res_b[r_inn2_slice] -= img_r[:-4:2, 3:-2:2]   # 2 x Up
    res_b[r_inn2_slice] -= img_r[4::2, 3:-2:2]    # 2 x Down
    res_b[r_inn2_slice] -= img_r[2:-2:2, 1:-4:2]  # 2 x Left
    res_b[r_inn2_slice] -= img_r[2:-2:2, 5::2]    # 2 x Right

    # Don't forget!
    img_r[r_inn0_slice] //= 3

    # 6.3 Add Blue
    res_b[r_inn2_slice] += img_b[1:-3:2, 2:-3:2]  # Left  Up
    res_b[r_inn2_slice] += img_b[1:-3:2, 4:-1:2]  # Right Up
    res_b[r_inn2_slice] += img_b[3:-1:2, 2:-3:2]  # Left  Down
    res_b[r_inn2_slice] += img_b[3:-1:2, 4:-1:2]  # Right Down

    img_b[r_inn1_slice] >>= 2

    normilize_color_range(res_b, color_radius)

    res_b[g_inn2_slice_even] >>= 4
    res_b[g_inn2_slice_odd] >>= 4
    res_b[r_inn2_slice] >>= 4

    np.clip(res_b, 0, 255, out=res_b)

    return np.dstack((res_r.astype(np.uint8), res_g.astype(np.uint8), res_b.astype(np.uint8)))


def compute_psnr(img_pred, img_gt):
    dtype = np.float64

    img_gt, img_pred = img_gt.astype(dtype), img_pred.astype(dtype)

    down = np.sum((img_gt - img_pred) ** 2)
    if down < 1e-10:
        raise ValueError

    up = np.prod(img_pred.shape, dtype=dtype) * (np.max(np.abs(img_gt) ** 2))

    return 10 * np.log10(up / down)

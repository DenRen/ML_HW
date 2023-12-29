import numpy as np


def compute_energy(img: np.ndarray):
    # Y = 0.299*R + 0.587*G + 0.114*B
    img_R, img_G, img_B = img.transpose(2, 0, 1).astype(np.float64)
    gray_img = 0.299*img_R + 0.587*img_G + 0.114*img_B

    dx_gray_img = np.zeros_like(gray_img)
    dx_gray_img[:, :-1] = gray_img[:, 1:]
    dx_gray_img[:, 1:] -= gray_img[:, :-1]
    dx_gray_img[:, 1:-1] *= 0.5
    dx_gray_img[:, 0] -= gray_img[:, 0]
    dx_gray_img[:, -1] += gray_img[:, -1]
    np.power(dx_gray_img, 2, out=dx_gray_img)

    dy_gray_img = np.zeros_like(gray_img)
    dy_gray_img[:-1, :] = gray_img[1:, :]
    dy_gray_img[1:, :] -= gray_img[:-1, :]
    dy_gray_img[1:-1, :] *= 0.5
    dy_gray_img[0, :] -= gray_img[0, :]
    dy_gray_img[-1, :] += gray_img[-1, :]
    np.power(dy_gray_img, 2, out=dy_gray_img)

    dy_gray_img += dx_gray_img
    np.sqrt(dy_gray_img, out=dy_gray_img)
    return dy_gray_img


def compute_seam_matrix(energy, mode, mask=None):
    if mask is None:
        seam = energy.copy()
    else:
        seam = mask.astype(np.float64)
        seam *= 256 * energy.shape[0] * energy.shape[1]
        seam += energy

    match mode:
        case 'vertical':
            for i_col in range(1, seam.shape[1]):
                seam[0, i_col] += np.min(seam[:2, i_col-1])
                seam[1:-1, i_col] += np.min([
                    seam[:-2, i_col-1],
                    seam[1:-1, i_col-1],
                    seam[2:, i_col-1]
                ], axis=0)
                seam[-1, i_col] += np.min(seam[-2:, i_col-1])
        case 'horizontal':
            for i_row in range(1, seam.shape[0]):
                seam[i_row, 0] += np.min(seam[i_row-1, :2])
                seam[i_row, 1:-1] += np.min([
                    seam[i_row-1, :-2],
                    seam[i_row-1, 1:-1],
                    seam[i_row-1, 2:]
                ], axis=0)
                seam[i_row, -1] += np.min(seam[i_row-1, -2:])
        case _:
            raise RuntimeError('Incorrect mode')

    return seam


def remove_minimal_seam(image, seam_matrix, mode, mask=None):
    n_rows, n_cols = seam_matrix.shape

    seam_mask = np.zeros_like(seam_matrix, dtype=np.uint8)
    match mode:
        case 'vertical shrink':
            img = image[:-1].astype(np.uint8)
            new_mask = None if mask is None else mask[:-1].copy()

            view_begin, view_end = 0, n_rows
            for i_col in range(n_cols-1, -1, -1):
                min_pos = np.argmin(
                    seam_matrix[view_begin:view_end, i_col])
                min_pos += view_begin

                seam_mask[min_pos, i_col] = 1
                view_begin = max(0, min_pos - 1)
                view_end = min(n_rows, min_pos + 2)

                img[min_pos:, i_col] = image[min_pos+1:, i_col]
                if mask is not None:
                    new_mask[min_pos:, i_col] = mask[min_pos+1:, i_col]
        case 'horizontal shrink':
            img = image[:, :-1].astype(np.uint8)
            new_mask = None if mask is None else mask[:, :-1].copy()

            view_begin, view_end = 0, n_cols
            for i_row in range(n_rows-1, -1, -1):
                min_pos = np.argmin(
                    seam_matrix[i_row, view_begin:view_end])
                min_pos += view_begin

                seam_mask[i_row, min_pos] = 1
                view_begin = max(0, min_pos - 1)
                view_end = min(n_cols, min_pos + 2)

                img[i_row, min_pos:] = image[i_row, min_pos+1:]

                if mask is not None:
                    new_mask[i_row, min_pos:] = mask[i_row, min_pos+1:]
        case _:
            raise RuntimeError('Incorrect mode')

    return img, new_mask, seam_mask


def seam_carve(img, mode, mask):
    energy = compute_energy(img)
    seam = compute_seam_matrix(energy, mode.split()[0], mask)
    img, new_mask, seam_mask = remove_minimal_seam(img, seam, mode, mask)
    return img, new_mask, seam_mask

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
from numpy.linalg import eigh

# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    matrix = matrix.copy()

    N = matrix.shape[0]
    assert N == matrix.shape[1]

    # Отцентруем каждую строчку матрицы
    mean_matrix = np.mean(matrix, axis=1)
    matrix -= mean_matrix[:, np.newaxis]

    # Найдем матрицу ковариации
    cov = np.zeros_like(matrix)
    for i in range(0, N):
        for j in range(i, N):
            cov[j, i] = cov[i, j] = matrix[i].dot(matrix[j]) / N

    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eigen_values, eigen_vectors = eigh(cov)

    # Посчитаем количество найденных собственных векторов
    # Оставляем только p собственных векторов
    eigen_ids = np.abs(eigen_values).argsort()[-p:][::-1]

    # Сортируем собственные значения в порядке убывания
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    eigen_vectors = eigen_vectors[:, eigen_ids]

    # Проекция данных на новое пространство
    return eigen_vectors, eigen_vectors.T @ matrix, mean_matrix


def pca_decompression(compressed):
    """Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    N = compressed[0][0].shape[0]
    result_img = np.zeros((N, N, 3), dtype=np.uint8)
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!

        eigen_vectors, matrix, mean_matrix = comp
        img = eigen_vectors @ matrix + mean_matrix[:, np.newaxis]
        result_img[..., i] = img.clip(0.0, 255.0).astype(np.uint8)

    return result_img


def pca_visualize():
    plt.clf()
    img = imread("cat.jpg")
    if len(img.shape) == 3:
        img = img[..., :3]  # WTF? may be > 3 ?
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    img = np.array((img[..., 0], img[..., 1], img[..., 2]), dtype=np.float32)
    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = [pca_compression(img[c], p) for c in range(3)]
        decompressed = pca_decompression(compressed)

        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title("Компонент: {}".format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    rot = np.array(
        [[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]]
    )
    shift = np.array([[0], [128], [128]])

    N, M = img.shape[:2]
    color_lines = shift + rot @ img.transpose(2, 0, 1).reshape(3, -1)
    return color_lines.reshape(3, N, M).transpose(1, 2, 0)


def ycbcr2rgb(img):
    """Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """

    rot = np.array([[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.77, 0]])

    N, M = img.shape[:2]

    conv_img = img.transpose(2, 0, 1)
    conv_img[1:3] -= 128

    color_lines = rot @ conv_img.reshape(3, -1)
    color_lines = color_lines.clip(0.0, 255.0).astype(np.uint8)
    return color_lines.reshape(3, N, M).transpose(1, 2, 0)


_SIGMA = 10.0


def get_gauss_1():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr_img = rgb2ycbcr(rgb_img)
    for i in range(1, 3):
        ycbcr_img[..., i] = gaussian_filter(ycbcr_img[..., i], _SIGMA)
    img = ycbcr2rgb(ycbcr_img)

    plt.imshow(img)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr_img = rgb2ycbcr(rgb_img)
    ycbcr_img[..., 0] = gaussian_filter(ycbcr_img[..., 0], _SIGMA)
    img = ycbcr2rgb(ycbcr_img)

    plt.imshow(img)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """

    gaussian_filter(component, _SIGMA, output=component)
    return component[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    assert block.shape == (8, 8)

    G = np.zeros_like(block, dtype=np.float64)

    r = np.arange(8)
    cos_table = np.cos(np.outer(r, (2 * r + 1) * (np.pi / 16)))

    for u in range(8):
        for v in range(8):
            G[u, v] = 0.25 * (block * np.outer(cos_table[u, :], cos_table[v, :])).sum()

    coef = 1 / np.sqrt(2)
    G[0, :] *= coef
    G[:, 0] *= coef

    return G


# Матрица квантования яркости
y_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

# Матрица квантования цвета
color_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    S = 5000.0 / q if q < 50 else (200.0 - 2 * q) if q < 100 else 1.0

    Q = (50 + S * default_quantization_matrix) * 0.01
    Q = Q.astype(np.int32)
    Q.clip(1, out=Q)
    return Q



zigzag_idxs = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]

inv_zigzag_idxs = [
     0,  1,  5,  6, 14, 15, 27, 28, 
     2,  4,  7, 13, 16, 26, 29, 42, 
     3,  8, 12, 17, 25, 30, 41, 43, 
     9, 11, 18, 24, 31, 40, 44, 53, 
    10, 19, 23, 32, 39, 45, 52, 54, 
    20, 22, 33, 38, 46, 51, 55, 60, 
    21, 34, 37, 47, 50, 56, 59, 61, 
    35, 36, 48, 49, 57, 58, 62, 63, 
]

def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """

    return block.flatten()[zigzag_idxs]


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    zigzag_len = len(zigzag_list)
    tmp_arr = np.zeros(2 * zigzag_len)

    w_idx, r_idx = 0, 0
    while r_idx < zigzag_len:
        if zigzag_list[r_idx] != 0:
            tmp_arr[w_idx] = zigzag_list[r_idx]
            w_idx += 1
            r_idx += 1
        else:
            zero_ctr = 1
            tmp_arr[w_idx] = 0
            w_idx += 1
            r_idx += 1

            while r_idx < zigzag_len and zigzag_list[r_idx] == 0:
                zero_ctr += 1
                r_idx += 1

            tmp_arr[w_idx] = zero_ctr
            w_idx += 1

    return tmp_arr[:w_idx]


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Переходим из RGB в YCbCr
    y, Cb, Cr = rgb2ycbcr(img).transpose(2, 0, 1)

    # Уменьшаем цветовые компоненты
    Cb = downsampling(Cb)
    Cr = downsampling(Cr)

    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    def get_compressed(matrix, quant_matrix):
        def compress_block(block):
            G = dct(block - 128)
            G_quant = quantization(G, quant_matrix)
            G_zigzag = zigzag(G_quant)
            return compression(G_zigzag)

        N, M = matrix.shape[:3]
        return [
            compress_block(matrix[8*row : 8*row + 8, 8*col : 8*col + 8])
            for row in range(N // 8)
            for col in range(M // 8)
        ]

    y_compressed  = get_compressed( y, quantization_matrixes[0])
    Cb_compressed = get_compressed(Cb, quantization_matrixes[1])
    Cr_compressed = get_compressed(Cr, quantization_matrixes[1])

    return y_compressed, Cb_compressed, Cr_compressed


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    res_list = np.zeros(64, dtype=np.float32)
    
    size = len(compressed_list)
    r, w = 0, 0
    while r < size:
        if compressed_list[r] != 0:
            res_list[w] = compressed_list[r]
            w += 1
            r += 1
        else:
            if r == size - 1:
                w += 1
                break
            
            zero_count = int(compressed_list[r + 1])
            w += zero_count
            r += 2

    return res_list[:w]


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    return input[inv_zigzag_idxs].reshape(8, 8)


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    f = np.zeros_like(block, dtype=np.float32)

    r = np.arange(8)
    cos_table = np.cos(np.outer((2 * r + 1) * (np.pi / 16), r))

    coef = 1 / np.sqrt(2)
    for x in range(8):
        for y in range(8):
            M = np.outer(cos_table[x, :], cos_table[y, :])
            M[0, :] *= coef
            M[:, 0] *= coef
            f[x, y] = 0.25 * (block * M).sum()

    return np.round(f)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """

    N, M = component.shape[:3]
    res = np.zeros((2 * N, 2 * M), dtype=component.dtype)
    res[0::2, 0::2] = component
    res[0::2, 1::2] = component
    res[1::2, 0::2] = component
    res[1::2, 1::2] = component

    return res


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    N, M, _ = result_shape
    y_compressed, Cb_compressed, Cr_compressed = result
    
    y = np.zeros((N, M))
    Cb, Cr = np.zeros((2, N // 2, M // 2))
    
    def fill_decompressed(res, compressed, quant_matrix):
        def decompress_block(block):
            block = inverse_compression(block)
            block = inverse_zigzag(block)
            block = inverse_quantization(block, quant_matrix)
            block = inverse_dct(block)
            block = np.round(block + 128)
            return block.clip(0, 255).astype(np.uint8)

        x_size = res.shape[1] // 8
        for i, block in enumerate(compressed):
            y, x = i // x_size * 8, i % x_size * 8
            res[y : y + 8, x : x + 8] = decompress_block(block)

    fill_decompressed( y,  y_compressed, quantization_matrixes[0])
    fill_decompressed(Cb, Cb_compressed, quantization_matrixes[1])
    fill_decompressed(Cr, Cr_compressed, quantization_matrixes[1])
    
    Cb = upsampling(Cb)
    Cr = upsampling(Cr)
    
    ycbcr_img = np.array([y, Cb, Cr]).transpose(1, 2, 0)
    img = ycbcr2rgb(ycbcr_img)
    return img

def jpeg_visualize():
    plt.clf()
    img = imread("Lenna.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        quant_matrixes = []
        quant_matrixes.append(own_quantization_matrix(y_quantization_matrix, p))
        quant_matrixes.append(own_quantization_matrix(color_quantization_matrix, p))

        jpeg_compressed = jpeg_compression(img, quant_matrixes)
        jpeg_decompressed = jpeg_decompression(jpeg_compressed, img.shape, quant_matrixes)

        axes[i // 3, i % 3].imshow(jpeg_decompressed)
        axes[i // 3, i % 3].set_title(f"Quality Factor: {p}")

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    if c_type.lower() == "jpeg":
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == "pca":
        compressed = []
        for j in range(0, 3):
            compressed.append(
                (pca_compression(img[:, :, j].astype(np.float64).copy(), param))
            )

        img = pca_decompression(compressed)
        compressed.extend(
            [
                np.mean(img[:, :, 0], axis=1),
                np.mean(img[:, :, 1], axis=1),
                np.mean(img[:, :, 2], axis=1),
            ]
        )

    if "tmp" not in os.listdir() or not os.path.isdir("tmp"):
        os.mkdir("tmp")

    compressed = np.array(compressed, dtype=np.object_)
    np.savez_compressed(os.path.join("tmp", "tmp.npz"), compressed)
    size = os.stat(os.path.join("tmp", "tmp.npz")).st_size * 8
    os.remove(os.path.join("tmp", "tmp.npz"))

    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title("PSNR for {}".format(c_type.upper()))
    ax1.plot(param_list, psnr, "tab:orange")
    ax1.set_xlabel("Quality Factor")
    ax1.set_ylabel("PSNR")

    ax2.set_title("Rate-Distortion for {}".format(c_type.upper()))
    ax2.plot(psnr, rate, "tab:red")
    ax2.set_xlabel("Distortion")
    ax2.set_ylabel("Rate")
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "pca", [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "jpeg", [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()

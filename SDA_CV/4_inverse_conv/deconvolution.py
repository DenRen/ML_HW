import numpy as np
from scipy.fft import fft2, ifft2, ifftshift, fftshift

def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """

    X2 = np.square(np.arange(size) - (size - 1) / 2)
    gauss_X2 = np.exp(-0.5 / (sigma ** 2) * X2)
    gauss_R2 = np.outer(gauss_X2, gauss_X2)
    gauss = gauss_R2 / gauss_R2.sum()
    return gauss


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    
    th, tw = shape
    kh, kw = h.shape
    ph, pw = th - kh, tw - kw
    padding = [((ph+1) // 2, ph // 2), ((pw+1) // 2, pw // 2)]  
    
    return fft2(ifftshift(np.pad(h, padding)))


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    
    H_inv = np.zeros_like(H)
    is_nozero = np.abs(H) > threshold
    H_inv[is_nozero] = 1 / H[is_nozero]
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    shape = blurred_img.shape
    G = fourier_transform(blurred_img, shape)
    H = fourier_transform(h, shape)
    H_inv = inverse_kernel(H, threshold)
    return fftshift(np.abs(ifft2(G * H_inv)))


def wiener_filtering(blurred_img, h, K=0.000043):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    
    shape = blurred_img.shape
    H = fourier_transform(h, shape)
    H_conj = np.conj(H)
    G = fourier_transform(blurred_img, shape)
    F = H_conj / (H * H_conj + K) * G
    return fftshift(np.abs(ifft2(F)))


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    
    MAX_I = 255
    
    buf = (img1 - img2) ** 2
    mse = buf.sum() / (buf.shape[0] * buf.shape[1])
    return 20 * np.log10(MAX_I / np.sqrt(mse))
    

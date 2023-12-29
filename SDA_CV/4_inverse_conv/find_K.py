from deconvolution import (
    compute_psnr,
    gaussian_kernel,
    wiener_filtering,
)
from visualization import vis_wiener
import numpy as np
import os

def main():
    basedir = "examples"
    original_img = np.load(os.path.join(basedir, "original.npy"))
    noisy_img = np.load(os.path.join(basedir, "noisy.npy"))
    
    baseline_psnr = compute_psnr(original_img, noisy_img)
    print(f"Baseline: {baseline_psnr}")

    kernel = gaussian_kernel(size=15, sigma=5)
    
    def calc_psnr(K):
        filtered_img = wiener_filtering(noisy_img, kernel, K)
        return compute_psnr(original_img, filtered_img)
    
    arr = []
    for K in np.linspace(0.000028, 0.000081, 1000):
        psnr = calc_psnr(K)
        print(f"{K:0.6f} {psnr:2.3f}")
        arr.append([psnr, K])
    
    psnr, K = sorted(arr, reverse=True)[0]
    print(f"Best result: {K:0.6f} {psnr:2.3f}")
    
    if psnr - baseline_psnr > 7:
        print("Solved!")
    
    
        
    
    # psnr = -1
    # while psnr - baseline_psnr < 7:
    #     if K is None:
    #         K = 1
    #         direct = False        

        
        
    #     print(f"{K:1.3f} -> {psnr}")
        
        
    #     if psnr > max_psnr_K[0]:
    #         max_psnr_K = [psnr, K]
    
    # psnr, K = max_psnr_K
    # print(f"Result: {K} -> {psnr}")
    
if __name__ == "__main__":
    main()
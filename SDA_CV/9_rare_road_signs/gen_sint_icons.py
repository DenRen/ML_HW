from rare_traffic_sign_solution import SignGenerator, generate_all_data
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def open_icon(icon_path):
    with Image.open(icon_path) as img_file:
        img = img_file.convert("RGBA")
    return np.array(img, dtype=np.uint8)


background_path = "background_images"
icons_path = "icons"
icon_path = "icons/1.17.png"
output_folder = "sint_icons"

# print(icon_path.split("/")[-1].rsplit(".", 1)[0])

# gen = SignGenerator(background_path)
# icon = open_icon(icon_path)
# sint_icon = gen.get_sample(icon)

# print(np.min(img), np.max(img))
# print(img.shape)
# plt.imshow(sint_icon)
# plt.show()

# gen.get_sample()

generate_all_data(output_folder, icons_path, background_path)
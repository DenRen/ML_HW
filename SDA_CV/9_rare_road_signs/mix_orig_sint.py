import os
import shutil

from sys import argv, exit
from os import listdir
from os.path import join, exists

def print_nums(classes_dir="cropped-train"):
    for class_name in sorted(os.listdir(classes_dir)):
        path = os.path.join(classes_dir, class_name)
        num_imgs = len(os.listdir(path))
        print(f"{class_name}: {num_imgs}")

def main():
    min_class_size_max = 1000
    
    if len(argv) != 5:
        print(f"{argv[0]} <orig> <sint> <dest> <min_class_size>")
        exit()
    
    orig_path, sint_path, dest_path, min_class_size = argv[1:]
    min_class_size = int(min_class_size)
    if min_class_size > min_class_size_max:
        print(f"min_class_size must be less then {min_class_size_max}")
        exit()
        
    if exists(dest_path):
        print(f"Directory {dest_path} already exist")
        exit()
    
    shutil.copytree(orig_path, dest_path)
    
    for class_name in sorted(listdir(dest_path)):
        class_dest_path = join(dest_path, class_name)
        num_imgs = len(listdir(class_dest_path))
        print(f"{class_name}: {num_imgs}", end=" -> ")
        
        num_sint_imgs = max(0, min_class_size - num_imgs)
        if num_sint_imgs > 0:
            class_sint_path = join(sint_path, class_name)
            sint_imgs = sorted(listdir(class_sint_path))
            sint_imgs = sint_imgs[:num_sint_imgs]
            
            for sint_img in sint_imgs:
                sint_img_path = join(class_sint_path, sint_img)
                sint_img_path_dest = join(class_dest_path, sint_img)
                
                if exists(sint_img_path_dest):
                    raise RuntimeError(f"Already exists! {sint_img_path_dest}")
                
                shutil.copy(sint_img_path, class_dest_path)

        print(min_class_size)

if __name__ == "__main__":
    main()

import os
import torch
import sys
sys.path.append("D:\CNBNet")
import basicsr
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding

def process_image(img_path, output_path, model):
    file_client = FileClient('disk')
    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("Image path {} not working".format(img_path))

    img = img2tensor(img, bgr2rgb=True, float32=True)
    model.single_image_inference(img, output_path)
'''
def main():
    opt = parse_options(is_train=False)
    model = create_model(opt)
    

    input_dir = opt['img_path'].get('input_img')  # 输入目录
    output_dir = opt['img_path'].get('output_img')  # 输出目录
    print(f"input_dir: {input_dir}")
    print(f"img_name: {img_name}")
    # 遍历输入目录中的所有图像
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        process_image(img_path, output_path, model)
        print('Inference for {} finished.'.format(img_path))

if __name__ == '__main__':
    main()'''
def main():
    opt = parse_options(is_train=False)
    model = create_model(opt)

    # 这里是从配置文件中获取输入和输出目录
    input_dir = opt['img_path'].get('input_img')  # 输入目录
    output_dir = opt['img_path'].get('output_img')  # 输出目录

    # 打印目录来确认它们已被正确赋值
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")

    # 确保输入和输出目录不是None
    if not input_dir or not output_dir:
        raise ValueError("Input or output directory is not set properly in the configuration file.")

    # 遍历输入目录中的所有图像
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        process_image(img_path, output_path, model)
        print('Inference for {} finished.'.format(img_path))

if __name__ == '__main__':
    main()

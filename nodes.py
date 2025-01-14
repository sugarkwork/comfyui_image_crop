import os
import numpy as np
import torch
from PIL import Image


def convert_to_pil(image: torch.Tensor) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(np.clip(255. * image.squeeze(), 0, 255).astype(np.uint8))
    if isinstance(image, torch.Tensor):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    raise ValueError(f"Unknown image type: {type(image)}")


def convert_to_tensor(image: Image.Image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, np.ndarray):
        return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class CropInfo:
    def __init__(self, x, y, width, height, original_width, original_height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.original_width = original_width
        self.original_height = original_height

        print(f"crop_info: x={x}, y={y}, width={width}, height={height}, original_width={original_width}, original_height={original_height}")



class ExpandMultiple:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "target_size": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 128}),
                "multiple": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 32}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "CROP_INFO")
    RETURN_NAMES = ("image", "crop_info")

    FUNCTION = "expand"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def expand(self, image:torch.Tensor, target_size:int=1024, multiple:int=128):
        img = convert_to_pil(image).convert("RGBA")

        original_width, original_height = img.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            new_width = target_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(new_height * aspect_ratio)

        resized_image = img.resize((new_width, new_height), Image.LANCZOS)

        new_width_crop = new_width
        new_height_crop = new_height

        if new_width == target_size:
            new_height_crop = (new_height // multiple) * multiple
            new_height_crop += multiple
        else:
            new_width_crop = (new_width // multiple) * multiple
            new_width_crop += multiple

        processed_image = resized_image.crop((0, 0, new_width_crop, new_height))

        return (convert_to_tensor(processed_image), CropInfo(0, 0, new_width_crop, new_height_crop, new_width, new_height))


class CropTransparent:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "margin": ("FLOAT", {"default": 3.0, "min": 0.0, "step": 0.1}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "CROP_INFO")
    RETURN_NAMES = ("image", "crop_info")

    FUNCTION = "crop"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def crop(self, image:torch.Tensor, margin:float=3.0, threshold:float=0.0):
        img = convert_to_pil(image).convert("RGBA")
        alpha = np.array(img.split()[-1])
        
        # 透明でない部分を見つける
        non_transparent = alpha > threshold
        
        # バウンディングボックスを取得
        bbox = Image.fromarray(non_transparent).getbbox()
        
        if bbox:
            # 画像の寸法を取得
            width, height = img.size
            
            # マージンを計算（画像の短辺の指定された割合）
            margin = int(min(width, height) * (margin / 100))
            
            # バウンディングボックスを拡大
            bbox_with_margin = (
                max(0, bbox[0] - margin),
                max(0, bbox[1] - margin),
                min(width, bbox[2] + margin),
                min(height, bbox[3] + margin)
            )
            
            cropped_img = img.crop(bbox_with_margin)
        return (convert_to_tensor(cropped_img), CropInfo(*bbox_with_margin, width, height))



class RestoreCrop:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_info": ("CROP_INFO",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "restore"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def restore(self, image:torch.Tensor, crop_info:CropInfo):
        img = convert_to_pil(image).convert("RGBA")
        print(f"image: {img.size}") 

        restore_img = Image.new("RGBA", (crop_info.original_width, crop_info.original_height))
        img = img.resize((crop_info.width - crop_info.x, crop_info.height - crop_info.y))
        restore_img.paste(img, (crop_info.x, crop_info.y))

        print(f"restore: x={crop_info.x}, y={crop_info.y}, width={crop_info.width}, height={crop_info.height}, original_width={crop_info.original_width}, original_height={crop_info.original_height}")

        return (convert_to_tensor(restore_img), )


NODE_CLASS_MAPPINGS = {
    "CropTransparent": CropTransparent,
    "RestoreCrop": RestoreCrop,
    "ExpandMultiple": ExpandMultiple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropTransparent": "CropTransparent",
    "RestoreCrop": "RestoreCrop",
    "ExpandMultiple": "ExpandMultiple",
}


def simple_test():
    pass


#if __name__ == "__main__":
#    simple_test()
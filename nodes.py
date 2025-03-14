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

    def _new_size(self, original_width:int, original_height:int, target_size:int) -> tuple:
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            new_width = target_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(new_height * aspect_ratio)
        
        return (new_width, new_height)
    
    def _new_crop_size(self, new_width:int, new_height:int, multiple:int, target_size:int) -> tuple:
        new_width_crop = new_width
        new_height_crop = new_height
        if new_width == target_size:
            new_height_crop = (new_height // multiple) * multiple
            new_height_crop += multiple
        else:
            new_width_crop = (new_width // multiple) * multiple
            new_width_crop += multiple
        
        return (new_width_crop, new_height_crop)

    def expand(self, image:torch.Tensor, target_size:int=1024, multiple:int=128):
        img = convert_to_pil(image).convert("RGBA")

        original_width, original_height = img.size

        new_width, new_height = self._new_size(original_width, original_height, target_size)
        resized_image = img.resize((new_width, new_height), Image.LANCZOS)

        new_width_crop, new_height_crop = self._new_crop_size(new_width, new_height, multiple, target_size)
        processed_image = resized_image.crop((0, 0, new_width_crop, new_height_crop))

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
    
        # 画像の寸法を取得
        width, height = img.size

        if bbox:
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
        else:
            bbox_with_margin = (0, 0, width, height)
            cropped_img = img

        return (convert_to_tensor(cropped_img), CropInfo(*bbox_with_margin, width, height))


class CropReapply:
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

    FUNCTION = "reapply"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def reapply(self, image:torch.Tensor, crop_info:CropInfo):
        img = convert_to_pil(image).convert("RGBA")
        img = img.crop((crop_info.x, crop_info.y, crop_info.x + crop_info.width, crop_info.y + crop_info.height))
        return (convert_to_tensor(img), )


class RestoreCrop:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_info": ("CROP_INFO",)
            },
            "optional": {
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.1})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "restore"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def restore(self, image:torch.Tensor, crop_info:CropInfo, scale_by:float=1.0):
        img = convert_to_pil(image).convert("RGBA")

        restore_img = Image.new("RGBA", 
                                (int(crop_info.original_width * scale_by), int(crop_info.original_height * scale_by)), 
                                (0, 0, 0, 0))
        img = img.resize(
            (
                int(crop_info.width * scale_by) - int(crop_info.x * scale_by), 
                int(crop_info.height * scale_by) - int(crop_info.y * scale_by)
                ),
                Image.LANCZOS)
        restore_img.paste(img, (int(crop_info.x * scale_by), int(crop_info.y * scale_by)))

        return (convert_to_tensor(restore_img), )


NODE_CLASS_MAPPINGS = {
    "CropTransparent": CropTransparent,
    "RestoreCrop": RestoreCrop,
    "ExpandMultiple": ExpandMultiple,
    "CropReapply": CropReapply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropTransparent": "CropTransparent",
    "RestoreCrop": "RestoreCrop",
    "ExpandMultiple": "ExpandMultiple",
    "CropReapply": "CropReapply",
}


def simple_test():
    pass


#if __name__ == "__main__":
#    simple_test()

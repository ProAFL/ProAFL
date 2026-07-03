from PIL import Image

                                                                                  
                                                           

if hasattr(Image, "Resampling"):
    BICUBIC = Image.Resampling.BICUBIC
    BILINEAR = Image.Resampling.BILINEAR
    LINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST

    AFFINE = Image.Transform.AFFINE
    FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = Image.Transpose.FLIP_TOP_BOTTOM
    PERSPECTIVE = Image.Transform.PERSPECTIVE
else:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST
    LINEAR = Image.LINEAR

    AFFINE = Image.AFFINE
    FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = Image.FLIP_TOP_BOTTOM
    PERSPECTIVE = Image.PERSPECTIVE

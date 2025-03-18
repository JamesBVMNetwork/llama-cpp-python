from __future__ import annotations

import os
from ctypes import (
    c_bool,
    c_char_p,
    c_int,
    c_int32,  # Added for int32_t
    c_uint8,
    c_float,
    c_void_p,
    c_size_t,  # Added for size_t
    POINTER,
    _Pointer,
    Structure,
    byref,    # Added for pointer passing
)
import pathlib
from typing import (
    Union,
    NewType,
    Optional,
    TYPE_CHECKING,
    List,     # Added for array returns
)

import llama_cpp.llama_cpp as llama_cpp

from llama_cpp._ctypes_extensions import (
    load_shared_library,
    ctypes_function_for_shared_library,
)

if TYPE_CHECKING:
    from llama_cpp._ctypes_extensions import (
        CtypesArray,
    )

# Library loading remains unchanged
_libllava_base_name = "llava"
_libllava_override_path = os.environ.get("LLAVA_CPP_LIB")
_libllava_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib" if _libllava_override_path is None else pathlib.Path()
_libllava = load_shared_library(_libllava_base_name, _libllava_base_path)

ctypes_function = ctypes_function_for_shared_library(_libllava)

# Existing llava.h bindings remain unchanged (omitted for brevity)
# ...

################################################
# clip.h
################################################

# Type definitions
clip_ctx_p = NewType("clip_ctx_p", int)
clip_ctx_p_ctypes = c_void_p

# New structures
class clip_context_params(Structure):
    _fields_ = [
        ("use_gpu", c_bool),
        ("verbosity", c_int),
    ]

class clip_image_size(Structure):
    _fields_ = [
        ("width", c_int),
        ("height", c_int),
    ]

# Opaque structures (fields not exposed in clip.h)
class clip_image_u8(Structure):
    pass

class clip_image_f32(Structure):
    pass

class clip_image_u8_batch(Structure):
    _fields_ = [
        ("data", POINTER(clip_image_u8)),
        ("size", c_size_t),
    ]

class clip_image_f32_batch(Structure):
    _fields_ = [
        ("data", POINTER(clip_image_f32)),
        ("size", c_size_t),
    ]

# Existing function: clip_model_load (kept for compatibility, marked deprecated)
@ctypes_function("clip_model_load", [c_char_p, c_int], clip_ctx_p_ctypes)
def clip_model_load(fname: bytes, verbosity: Union[c_int, int], /) -> Optional[clip_ctx_p]:
    """Load a CLIP model (deprecated, use clip_init instead)."""
    ...

# New function: clip_init
@ctypes_function("clip_init", [c_char_p, clip_context_params], clip_ctx_p_ctypes)
def clip_init(fname: bytes, ctx_params: clip_context_params, /) -> Optional[clip_ctx_p]:
    """Initialize a CLIP context with parameters."""
    ...

# Existing function: clip_free
@ctypes_function("clip_free", [clip_ctx_p_ctypes], None)
def clip_free(ctx: clip_ctx_p, /):
    """Free a CLIP context."""
    ...

# New functions
@ctypes_function("clip_embd_nbytes", [clip_ctx_p_ctypes], c_size_t)
def clip_embd_nbytes(ctx: clip_ctx_p, /) -> int:
    """Get the number of bytes in the embedding."""
    ...

@ctypes_function("clip_embd_nbytes_by_img", [clip_ctx_p_ctypes, c_int, c_int], c_size_t)
def clip_embd_nbytes_by_img(ctx: clip_ctx_p, img_h: int, img_w: int, /) -> int:
    """Get the number of bytes in the embedding for an image of given height and width."""
    ...

@ctypes_function("clip_image_size", [clip_ctx_p_ctypes], c_int32)
def clip_image_size(ctx: clip_ctx_p, /) -> int:
    """Get the image size."""
    ...

@ctypes_function("clip_patch_size", [clip_ctx_p_ctypes], c_int32)
def clip_patch_size(ctx: clip_ctx_p, /) -> int:
    """Get the patch size."""
    ...

@ctypes_function("clip_hidden_size", [clip_ctx_p_ctypes], c_int32)
def clip_hidden_size(ctx: clip_ctx_p, /) -> int:
    """Get the hidden size."""
    ...

@ctypes_function("clip_patch_merge_type", [clip_ctx_p_ctypes], c_char_p)
def clip_patch_merge_type(ctx: clip_ctx_p, /) -> bytes:
    """Get the patch merge type as bytes."""
    ...

@ctypes_function("clip_image_grid", [clip_ctx_p_ctypes], POINTER(c_int32))
def clip_image_grid(ctx: clip_ctx_p, /) -> "_Pointer[c_int32]":
    """Get a pointer to the image grid array."""
    ...

@ctypes_function("get_clip_image_grid_size", [clip_ctx_p_ctypes], c_size_t)
def get_clip_image_grid_size(ctx: clip_ctx_p, /) -> int:
    """Get the size of the image grid."""
    ...

def get_clip_image_grid(ctx: clip_ctx_p, /) -> List[int]:
    """Get the image grid as a list."""
    size = get_clip_image_grid_size(ctx)
    grid_ptr = clip_image_grid(ctx)
    return [grid_ptr[i] for i in range(size)]

@ctypes_function("clip_n_patches", [clip_ctx_p_ctypes], c_int)
def clip_n_patches(ctx: clip_ctx_p, /) -> int:
    """Get the number of patches."""
    ...

@ctypes_function("clip_n_patches_by_img", [clip_ctx_p_ctypes, POINTER(clip_image_f32)], c_int)
def clip_n_patches_by_img(ctx: clip_ctx_p, img: "_Pointer[clip_image_f32]", /) -> int:
    """Get the number of patches for a given image."""
    ...

@ctypes_function("clip_n_mmproj_embd", [clip_ctx_p_ctypes], c_int)
def clip_n_mmproj_embd(ctx: clip_ctx_p, /) -> int:
    """Get the number of multimodal projection embeddings."""
    ...

@ctypes_function("clip_uhd_num_image_embeds_col", [clip_ctx_p_ctypes], c_int)
def clip_uhd_num_image_embeds_col(ctx_clip: clip_ctx_p, /) -> int:
    """Get the number of UHD image embeds per column."""
    ...

@ctypes_function("clip_add_load_image_size", [clip_ctx_p_ctypes, POINTER(clip_image_size)], None)
def clip_add_load_image_size(ctx_clip: clip_ctx_p, load_image_size: "_Pointer[clip_image_size]", /):
    """Add a load image size to the context."""
    ...

@ctypes_function("clip_get_load_image_size", [clip_ctx_p_ctypes], POINTER(clip_image_size))
def clip_get_load_image_size(ctx_clip: clip_ctx_p, /) -> "_Pointer[clip_image_size]":
    """Get the load image size from the context."""
    ...

@ctypes_function("clip_image_size_init", [], POINTER(clip_image_size))
def clip_image_size_init(/) -> "_Pointer[clip_image_size]":
    """Initialize a clip_image_size structure."""
    ...

@ctypes_function("clip_image_u8_init", [], POINTER(clip_image_u8))
def clip_image_u8_init(/) -> "_Pointer[clip_image_u8]":
    """Initialize a clip_image_u8 structure."""
    ...

@ctypes_function("clip_image_f32_init", [], POINTER(clip_image_f32))
def clip_image_f32_init(/) -> "_Pointer[clip_image_f32]":
    """Initialize a clip_image_f32 structure."""
    ...

@ctypes_function("clip_image_u8_free", [POINTER(clip_image_u8)], None)
def clip_image_u8_free(img: "_Pointer[clip_image_u8]", /):
    """Free a clip_image_u8 structure."""
    ...

@ctypes_function("clip_image_f32_free", [POINTER(clip_image_f32)], None)
def clip_image_f32_free(img: "_Pointer[clip_image_f32]", /):
    """Free a clip_image_f32 structure."""
    ...

@ctypes_function("clip_image_u8_batch_free", [POINTER(clip_image_u8_batch)], None)
def clip_image_u8_batch_free(batch: "_Pointer[clip_image_u8_batch]", /):
    """Free a clip_image_u8_batch structure."""
    ...

@ctypes_function("clip_image_f32_batch_free", [POINTER(clip_image_f32_batch)], None)
def clip_image_f32_batch_free(batch: "_Pointer[clip_image_f32_batch]", /):
    """Free a clip_image_f32_batch structure."""
    ...

@ctypes_function("clip_build_img_from_pixels", [POINTER(c_uint8), c_int, c_int, POINTER(clip_image_u8)], None)
def clip_build_img_from_pixels(rgb_pixels: "CtypesArray[c_uint8]", nx: int, ny: int, img: "_Pointer[clip_image_u8]", /):
    """Build an image from RGB pixels."""
    ...

@ctypes_function("clip_image_load_from_file", [c_char_p, POINTER(clip_image_u8)], c_bool)
def clip_image_load_from_file(fname: bytes, img: "_Pointer[clip_image_u8]", /) -> bool:
    """Load an image from a file into a clip_image_u8 structure."""
    ...

@ctypes_function("clip_image_load_from_bytes", [POINTER(c_uint8), c_size_t, POINTER(clip_image_u8)], c_bool)
def clip_image_load_from_bytes(bytes: "CtypesArray[c_uint8]", bytes_length: int, img: "_Pointer[clip_image_u8]", /) -> bool:
    """Load an image from bytes into a clip_image_u8 structure."""
    ...

@ctypes_function("clip_image_preprocess", [clip_ctx_p_ctypes, POINTER(clip_image_u8), POINTER(clip_image_f32_batch)], c_bool)
def clip_image_preprocess(ctx: clip_ctx_p, img: "_Pointer[clip_image_u8]", res_imgs: "_Pointer[clip_image_f32_batch]", /) -> bool:
    """Preprocess an image and store results in a batch."""
    ...

@ctypes_function("clip_get_newline_tensor", [clip_ctx_p_ctypes], c_void_p)  # Assuming ggml_tensor * as c_void_p
def clip_get_newline_tensor(ctx: clip_ctx_p, /) -> c_void_p:
    """Get the newline tensor."""
    ...

@ctypes_function("clip_image_encode", [clip_ctx_p_ctypes, c_int, POINTER(clip_image_f32), POINTER(c_float)], c_bool)
def clip_image_encode(ctx: clip_ctx_p, n_threads: int, img: "_Pointer[clip_image_f32]", vec: "_Pointer[c_float]", /) -> bool:
    """Encode an image into a float vector."""
    ...

@ctypes_function("clip_image_batch_encode", [clip_ctx_p_ctypes, c_int, POINTER(clip_image_f32_batch), POINTER(c_float)], c_bool)
def clip_image_batch_encode(ctx: clip_ctx_p, n_threads: int, imgs: "_Pointer[clip_image_f32_batch]", vec: "_Pointer[c_float]", /) -> bool:
    """Encode a batch of images into a float vector."""
    ...

@ctypes_function("clip_model_quantize", [c_char_p, c_char_p, c_int], c_bool)
def clip_model_quantize(fname_inp: bytes, fname_out: bytes, itype: int, /) -> bool:
    """Quantize a CLIP model."""
    ...

@ctypes_function("clip_is_minicpmv", [clip_ctx_p_ctypes], c_int)
def clip_is_minicpmv(ctx: clip_ctx_p, /) -> int:
    """Check if the model is MiniCPM-V."""
    ...

@ctypes_function("clip_is_glm", [clip_ctx_p_ctypes], c_bool)
def clip_is_glm(ctx: clip_ctx_p, /) -> bool:
    """Check if the model is GLM."""
    ...

@ctypes_function("clip_is_qwen2vl", [clip_ctx_p_ctypes], c_bool)
def clip_is_qwen2vl(ctx: clip_ctx_p, /) -> bool:
    """Check if the model is Qwen2-VL."""
    ...

@ctypes_function("get_deepest_feature_layer", [clip_ctx_p_ctypes], c_int)
def get_deepest_feature_layer(ctx: clip_ctx_p, /) -> int:
    """Get the deepest feature layer."""
    ...

@ctypes_function("clip_encode_float_image", [clip_ctx_p_ctypes, c_int, POINTER(c_float), c_int, c_int, POINTER(c_float)], c_bool)
def clip_encode_float_image(ctx: clip_ctx_p, n_threads: int, img: "_Pointer[c_float]", h: int, w: int, vec: "_Pointer[c_float]", /) -> bool:
    """Encode a float image into a vector."""
    ...
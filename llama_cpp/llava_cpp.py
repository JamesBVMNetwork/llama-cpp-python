from __future__ import annotations

import os
from ctypes import (
    c_bool,
    c_char_p,
    c_int,
    c_uint8,
    c_float,
    c_void_p,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
)
import pathlib
from typing import (
    Union,
    NewType,
    Optional,
    TYPE_CHECKING,
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


# Specify the base name of the shared library to load
_libllava_base_name = "llava"
_libllava_override_path = os.environ.get("LLAVA_CPP_LIB")
_libllava_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib" if _libllava_override_path is None else pathlib.Path()

# Load the library
_libllava = load_shared_library(_libllava_base_name, _libllava_base_path)

ctypes_function = ctypes_function_for_shared_library(_libllava)


################################################
# llava.h
################################################

# struct clip_ctx;
clip_ctx_p = NewType("clip_ctx_p", int)
clip_ctx_p_ctypes = c_void_p


# struct llava_image_embed {
#     float * embed;
#     int n_image_pos;
# };
class llava_image_embed(Structure):
    _fields_ = [
        ("embed", POINTER(c_float)),
        ("n_image_pos", c_int),
    ]


# /** sanity check for clip <-> llava embed size match */
# LLAVA_API bool llava_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip);
@ctypes_function(
    "llava_validate_embed_size",
    [llama_cpp.llama_context_p_ctypes, clip_ctx_p_ctypes],
    c_bool,
)
def llava_validate_embed_size(
    ctx_llama: llama_cpp.llama_context_p, ctx_clip: clip_ctx_p, /
) -> bool:
    ...


# /** build an image embed from image file bytes */
# LLAVA_API struct llava_image_embed * llava_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length);
@ctypes_function(
    "llava_image_embed_make_with_bytes",
    [clip_ctx_p_ctypes, c_int, POINTER(c_uint8), c_int],
    POINTER(llava_image_embed),
)
def llava_image_embed_make_with_bytes(
    ctx_clip: clip_ctx_p,
    n_threads: Union[c_int, int],
    image_bytes: CtypesArray[c_uint8],
    image_bytes_length: Union[c_int, int],
    /,
) -> "_Pointer[llava_image_embed]":
    ...


# /** build an image embed from a path to an image filename */
# LLAVA_API struct llava_image_embed * llava_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path);
@ctypes_function(
    "llava_image_embed_make_with_filename",
    [clip_ctx_p_ctypes, c_int, c_char_p],
    POINTER(llava_image_embed),
)
def llava_image_embed_make_with_filename(
    ctx_clip: clip_ctx_p, n_threads: Union[c_int, int], image_path: bytes, /
) -> "_Pointer[llava_image_embed]":
    ...


# LLAVA_API void llava_image_embed_free(struct llava_image_embed * embed);
# /** free an embedding made with llava_image_embed_make_* */
@ctypes_function("llava_image_embed_free", [POINTER(llava_image_embed)], None)
def llava_image_embed_free(embed: "_Pointer[llava_image_embed]", /):
    ...


# /** write the image represented by embed into the llama context with batch size n_batch, starting at context pos n_past. on completion, n_past points to the next position in the context after the image embed. */
# LLAVA_API bool llava_eval_image_embed(struct llama_context * ctx_llama, const struct llava_image_embed * embed, int n_batch, int * n_past);
@ctypes_function(
    "llava_eval_image_embed",
    [
        llama_cpp.llama_context_p_ctypes,
        POINTER(llava_image_embed),
        c_int,
        POINTER(c_int),
    ],
    c_bool,
)
def llava_eval_image_embed(
    ctx_llama: llama_cpp.llama_context_p,
    embed: "_Pointer[llava_image_embed]",
    n_batch: Union[c_int, int],
    n_past: "_Pointer[c_int]",
    /,
) -> bool:
    ...


################################################
# clip.h
################################################


import ctypes
from typing import Optional, Union

# Placeholder for ctypes_function decorator (assumed from original code)
def ctypes_function(name, argtypes, restype):
    def decorator(func):
        # Normally sets up ctypes.CFUNCTYPE and library loading
        return func
    return decorator

# Type aliases for clarity
clip_ctx_p = ctypes.c_void_p  # Pointer to clip_ctx (opaque struct)
c_char_p = ctypes.c_char_p
c_int = ctypes.c_int
c_bool = ctypes.c_bool
c_size_t = ctypes.c_size_t
c_int32 = ctypes.c_int32
c_float_p = ctypes.POINTER(ctypes.c_float)
c_uchar_p = ctypes.POINTER(ctypes.c_ubyte)

# Struct definitions
class clip_image_size(ctypes.Structure):
    _fields_ = [
        ("width", c_int),
        ("height", c_int),
    ]

clip_image_size_p = ctypes.POINTER(clip_image_size)

class clip_image_u8_batch(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),  # Pointer to clip_image_u8 (opaque)
        ("size", c_size_t),
    ]

class clip_image_f32_batch(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),  # Pointer to clip_image_f32 (opaque)
        ("size", c_size_t),
    ]

class clip_context_params(ctypes.Structure):
    _fields_ = [
        ("use_gpu", c_bool),
        ("verbosity", c_int),
    ]

# Opaque struct pointers
clip_image_u8_p = ctypes.c_void_p  # clip_image_u8 is opaque
clip_image_f32_p = ctypes.c_void_p  # clip_image_f32 is opaque
clip_ggml_tensor_p = ctypes.c_void_p  # ggml_tensor is opaque

# Deprecated clip_model_load
@ctypes_function("clip_model_load", [c_char_p, c_int], clip_ctx_p)
def clip_model_load(fname: bytes, verbosity: Union[c_int, int], /) -> Optional[clip_ctx_p]:
    """Deprecated: Load a CLIP model (use clip_init instead)."""
    pass

# clip_init
@ctypes_function("clip_init", [c_char_p, clip_context_params], clip_ctx_p)
def clip_init(fname: bytes, ctx_params: clip_context_params, /) -> Optional[clip_ctx_p]:
    """Initialize a CLIP context from a model file."""
    pass

# clip_free
@ctypes_function("clip_free", [clip_ctx_p], None)
def clip_free(ctx: clip_ctx_p, /) -> None:
    """Free a CLIP context."""
    pass

# clip_embd_nbytes
@ctypes_function("clip_embd_nbytes", [clip_ctx_p], c_size_t)
def clip_embd_nbytes(ctx: clip_ctx_p, /) -> int:
    """Get the size of embeddings in bytes."""
    pass

# clip_embd_nbytes_by_img
@ctypes_function("clip_embd_nbytes_by_img", [clip_ctx_p, c_int, c_int], c_size_t)
def clip_embd_nbytes_by_img(ctx: clip_ctx_p, img_h: int, img_w: int, /) -> int:
    """Get the size of embeddings in bytes for an image of given height and width."""
    pass

# clip_image_size
@ctypes_function("clip_image_size", [clip_ctx_p], c_int32)
def clip_image_size(ctx: clip_ctx_p, /) -> int:
    """Get the image size."""
    pass

# clip_patch_size
@ctypes_function("clip_patch_size", [clip_ctx_p], c_int32)
def clip_patch_size(ctx: clip_ctx_p, /) -> int:
    """Get the patch size."""
    pass

# clip_hidden_size
@ctypes_function("clip_hidden_size", [clip_ctx_p], c_int32)
def clip_hidden_size(ctx: clip_ctx_p, /) -> int:
    """Get the hidden size."""
    pass

# clip_patch_merge_type
@ctypes_function("clip_patch_merge_type", [clip_ctx_p], c_char_p)
def clip_patch_merge_type(ctx: clip_ctx_p, /) -> bytes:
    """Get the patch merge type as a string (returns bytes)."""
    pass

# clip_image_grid
@ctypes_function("clip_image_grid", [clip_ctx_p], ctypes.POINTER(c_int32))
def clip_image_grid(ctx: clip_ctx_p, /) -> ctypes.POINTER(c_int32):
    """Get the image grid as a pointer to int32."""
    pass

# get_clip_image_grid_size
@ctypes_function("get_clip_image_grid_size", [clip_ctx_p], c_size_t)
def get_clip_image_grid_size(ctx: clip_ctx_p, /) -> int:
    """Get the size of the image grid."""
    pass

# clip_n_patches
@ctypes_function("clip_n_patches", [clip_ctx_p], c_int)
def clip_n_patches(ctx: clip_ctx_p, /) -> int:
    """Get the number of patches."""
    pass

# clip_n_patches_by_img
@ctypes_function("clip_n_patches_by_img", [clip_ctx_p, clip_image_f32_p], c_int)
def clip_n_patches_by_img(ctx: clip_ctx_p, img: clip_image_f32_p, /) -> int:
    """Get the number of patches for a given image."""
    pass

# clip_n_mmproj_embd
@ctypes_function("clip_n_mmproj_embd", [clip_ctx_p], c_int)
def clip_n_mmproj_embd(ctx: clip_ctx_p, /) -> int:
    """Get the number of multimodal projection embeddings."""
    pass

# clip_uhd_num_image_embeds_col
@ctypes_function("clip_uhd_num_image_embeds_col", [clip_ctx_p], c_int)
def clip_uhd_num_image_embeds_col(ctx: clip_ctx_p, /) -> int:
    """Get the number of UHD image embeds per column."""
    pass

# clip_add_load_image_size
@ctypes_function("clip_add_load_image_size", [clip_ctx_p, ctypes.POINTER(clip_image_size)], None)
def clip_add_load_image_size(ctx: clip_ctx_p, load_image_size: clip_image_size_p, /) -> None:
    """Add load image size to the context."""
    pass

# clip_get_load_image_size
@ctypes_function("clip_get_load_image_size", [clip_ctx_p], clip_image_size_p)
def clip_get_load_image_size(ctx: clip_ctx_p, /) -> clip_image_size_p:
    """Get the load image size from the context."""
    pass

# clip_image_size_init
@ctypes_function("clip_image_size_init", [], clip_image_size_p)
def clip_image_size_init() -> clip_image_size_p:
    """Initialize a clip_image_size struct."""
    pass

# clip_image_u8_init
@ctypes_function("clip_image_u8_init", [], clip_image_u8_p)
def clip_image_u8_init() -> clip_image_u8_p:
    """Initialize a clip_image_u8 struct."""
    pass

# clip_image_f32_init
@ctypes_function("clip_image_f32_init", [], clip_image_f32_p)
def clip_image_f32_init() -> clip_image_f32_p:
    """Initialize a clip_image_f32 struct."""
    pass

# clip_image_u8_free
@ctypes_function("clip_image_u8_free", [clip_image_u8_p], None)
def clip_image_u8_free(img: clip_image_u8_p, /) -> None:
    """Free a clip_image_u8 struct."""
    pass

# clip_image_f32_free
@ctypes_function("clip_image_f32_free", [clip_image_f32_p], None)
def clip_image_f32_free(img: clip_image_f32_p, /) -> None:
    """Free a clip_image_f32 struct."""
    pass

# clip_image_u8_batch_free
@ctypes_function("clip_image_u8_batch_free", [ctypes.POINTER(clip_image_u8_batch)], None)
def clip_image_u8_batch_free(batch: ctypes.POINTER(clip_image_u8_batch), /) -> None:
    """Free a clip_image_u8_batch struct."""
    pass

# clip_image_f32_batch_free
@ctypes_function("clip_image_f32_batch_free", [ctypes.POINTER(clip_image_f32_batch)], None)
def clip_image_f32_batch_free(batch: ctypes.POINTER(clip_image_f32_batch), /) -> None:
    """Free a clip_image_f32_batch struct."""
    pass

# clip_build_img_from_pixels
@ctypes_function("clip_build_img_from_pixels", [c_uchar_p, c_int, c_int, clip_image_u8_p], None)
def clip_build_img_from_pixels(rgb_pixels: c_uchar_p, nx: int, ny: int, img: clip_image_u8_p, /) -> None:
    """Build an image from RGB pixels."""
    pass

# clip_image_load_from_file
@ctypes_function("clip_image_load_from_file", [c_char_p, clip_image_u8_p], c_bool)
def clip_image_load_from_file(fname: bytes, img: clip_image_u8_p, /) -> bool:
    """Load an image from a file into a clip_image_u8 struct."""
    pass

# clip_image_load_from_bytes
@ctypes_function("clip_image_load_from_bytes", [c_uchar_p, c_size_t, clip_image_u8_p], c_bool)
def clip_image_load_from_bytes(bytes_data: c_uchar_p, bytes_length: int, img: clip_image_u8_p, /) -> bool:
    """Load an image from bytes into a clip_image_u8 struct."""
    pass

# clip_image_preprocess
@ctypes_function("clip_image_preprocess", [clip_ctx_p, clip_image_u8_p, ctypes.POINTER(clip_image_f32_batch)], c_bool)
def clip_image_preprocess(ctx: clip_ctx_p, img: clip_image_u8_p, res_imgs: ctypes.POINTER(clip_image_f32_batch), /) -> bool:
    """Preprocess an image and store results in a clip_image_f32_batch."""
    pass

# clip_get_newline_tensor
@ctypes_function("clip_get_newline_tensor", [clip_ctx_p], clip_ggml_tensor_p)
def clip_get_newline_tensor(ctx: clip_ctx_p, /) -> clip_ggml_tensor_p:
    """Get the newline tensor."""
    pass

# clip_image_encode
@ctypes_function("clip_image_encode", [clip_ctx_p, c_int, clip_image_f32_p, c_float_p], c_bool)
def clip_image_encode(ctx: clip_ctx_p, n_threads: int, img: clip_image_f32_p, vec: c_float_p, /) -> bool:
    """Encode an image into a float vector."""
    pass

# clip_image_batch_encode
@ctypes_function("clip_image_batch_encode", [clip_ctx_p, c_int, ctypes.POINTER(clip_image_f32_batch), c_float_p], c_bool)
def clip_image_batch_encode(ctx: clip_ctx_p, n_threads: int, imgs: ctypes.POINTER(clip_image_f32_batch), vec: c_float_p, /) -> bool:
    """Encode a batch of images into a float vector."""
    pass

# clip_model_quantize
@ctypes_function("clip_model_quantize", [c_char_p, c_char_p, c_int], c_bool)
def clip_model_quantize(fname_inp: bytes, fname_out: bytes, itype: int, /) -> bool:
    """Quantize a CLIP model."""
    pass

# clip_is_minicpmv
@ctypes_function("clip_is_minicpmv", [clip_ctx_p], c_int)
def clip_is_minicpmv(ctx: clip_ctx_p, /) -> int:
    """Check if the model is MiniCPMV."""
    pass

# clip_is_glm
@ctypes_function("clip_is_glm", [clip_ctx_p], c_bool)
def clip_is_glm(ctx: clip_ctx_p, /) -> bool:
    """Check if the model is GLM."""
    pass

# clip_is_qwen2vl
@ctypes_function("clip_is_qwen2vl", [clip_ctx_p], c_bool)
def clip_is_qwen2vl(ctx: clip_ctx_p, /) -> bool:
    """Check if the model is Qwen2VL."""
    pass

# get_deepest_feature_layer
@ctypes_function("get_deepest_feature_layer", [clip_ctx_p], c_int)
def get_deepest_feature_layer(ctx: clip_ctx_p, /) -> int:
    """Get the deepest feature layer."""
    pass

# clip_encode_float_image
@ctypes_function("clip_encode_float_image", [clip_ctx_p, c_int, c_float_p, c_int, c_int, c_float_p], c_bool)
def clip_encode_float_image(ctx: clip_ctx_p, n_threads: int, img: c_float_p, h: int, w: int, vec: c_float_p, /) -> bool:
    """Encode a float image into a vector."""
    pass

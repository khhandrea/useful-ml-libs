# pip3 install einops
from einops import rearrange, reduce, repeat
import numpy as np

# Rearrange
x_image = np.random.rand(64, 24, 36, 3) # batch, row, column, dimension in OpenCV
x_tensor = rearrange(x_image, "b h w c -> b c w h")
x_flatten = rearrange(x_tensor, "b c w h -> b c (w h)")
x_flatten_expanded = rearrange(x_flatten, "b c wh -> b c 1 wh")
x_unflatten = rearrange(x_flatten, "b c (w h) -> b c w h", w=72)
print("x_image:", x_image.shape)
print("x_tensor:", x_tensor.shape)
print("x_flatten:", x_flatten.shape)
print("x_flatten_expanded:", x_flatten_expanded.shape)
print("x_unfllaten:", x_unflatten.shape)
print()

# Reduce
x_mean = reduce(x_tensor, "b c w h -> c w h", "mean")
x_max = reduce(x_tensor, "b c w h -> c w h", "max")
print("x_mean:", x_mean.shape)
print("x_max:", x_max.shape)
print()

# Repeat
x_one_image = np.random.rand(3, 36, 24)
x_dummy = repeat(x_one_image, "c w h -> 64 c w h")
x_repeated_dummy = repeat(x_dummy, "b c w h -> (4 b) c w h")
print("x_dummy_batch:", x_dummy.shape)
print("x_repeated_dummy:", x_repeated_dummy.shape)
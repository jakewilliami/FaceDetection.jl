#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
#=
Rectangle features can be computed very rapidly using an intermediate representation for the image, which we call the integral image.
The integral image at location $x,y$ contains the sum of the pixels above and to the left of $x,y$ inclusive.
Original    Integral
+--------   +------------
| 1 2 3 .   | 1  3  6 .
| 4 5 6 .   | 5 12 21 .
| . . . .   | . . . . .
=#

import Base: size, getindex, LinearIndices
using Images: Images, coords_spatial

struct IntegralArray{T, N, A} <: AbstractArray{T, N}
	data::A
end

#=
	to_integral_image(img_arr::AbstractArray) -> AbstractArray

Calculates the integral image based on this instance's original image data.

# Arguments

- `img_arr::AbstractArray`: Image source data

# Returns

 - `integral_image_arr::AbstractArray`: Integral image for given image
=#
function to_integral_image(img_arr::AbstractArray)
	array_size = size(img_arr)
    integral_image_arr = Array{Images.accum(eltype(img_arr))}(undef, array_size)
    sd = coords_spatial(img_arr)
    cumsum!(integral_image_arr, img_arr; dims=sd[1])#length(array_size)
    for i = 2:length(sd)
        cumsum!(integral_image_arr, integral_image_arr; dims=sd[i])
    end
	
    return Array{eltype(img_arr), ndims(img_arr)}(integral_image_arr)
end

LinearIndices(A::IntegralArray) = Base.LinearFast()
size(A::IntegralArray) = size(A.data)
getindex(A::IntegralArray, i::Int...) = A.data[i...]
getindex(A::IntegralArray, ids::Tuple...) = getindex(A, ids[1]...)

#=
	sum_region(
		integral_image_arr::AbstractArray,
		top_left::Tuple{Int64,Int64},
		bottom_right::Tuple{Int64,Int64}
	) -> Number

# Arguments

- `integral_image_arr::AbstractArray`: The intermediate Integral Image
- `top_left::Tuple{Integer, Integer}`: (x,y) of the rectangle's top left corner
- `bottom_right::Tuple{Integer, Integer}`: (x,y) of the rectangle's bottom right corner

# Returns

- `sum::Number` The sum of all pixels in the given rectangle defined by the parameters top_left and bottom_right
=#
function sum_region(
	integral_image_arr::AbstractArray,
	top_left::Tuple{Integer,Integer},
	bottom_right::Tuple{Integer,Integer}
)
	sum = integral_image_arr[bottom_right[2], bottom_right[1]]
    sum -= top_left[1] > 1 ? integral_image_arr[bottom_right[2], top_left[1] - 1] : zero(Int64)
    sum -= top_left[2] > 1 ? integral_image_arr[top_left[2] - 1, bottom_right[1]] : zero(Int64)
    sum += top_left[2] > 1 && top_left[1] > 1 ? integral_image_arr[top_left[2] - 1, top_left[1] - 1] : zero(Int64)
	
    return sum
end

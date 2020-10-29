#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
#=
	IntegralArray{T, N, A} <: AbstractArray{T, N}
	
Rectangle features can be computed very rapidly using an intermediate representation for the image, which we call the integral image.
The integral image at location $x,y$ contains the sum of the pixels above and to the left of $x,y$ inclusive.
Original    Integral
+--------   +------------
| 1 2 3 .   | 1  3  6 .
| 4 5 6 .   | 5 12 21 .
| . . . .   | . . . . .
=#
struct IntegralArray{T, N, A} <: AbstractArray{T, N}
	data::A
end

"""
	to_integral_image(img_arr::AbstractArray) -> AbstractArray

Calculates the integral image based on this instance's original image data.

# Arguments

- `img_arr::AbstractArray`: Image source data

# Returns

 - `integral_image_arr::AbstractArray`: Integral image for given image
 """
function to_integral_image(img_arr::AbstractArray{T, N}) where {T, N}
	array_size = size(img_arr)
    integral_image_arr = Array{Images.accum(eltype(img_arr))}(undef, array_size)
    sd = coords_spatial(img_arr)
    cumsum!(integral_image_arr, img_arr; dims=first(sd))
    for i = 2:length(sd)
        cumsum!(integral_image_arr, integral_image_arr; dims=sd[i])
    end
	
    return Array{T, N}(integral_image_arr)
end

LinearIndices(A::IntegralArray) = Base.LinearFast()
@inline size(A::IntegralArray) = size(A.data)
@inline getindex(A::IntegralArray, i::Int...) = A.data[i...]
@inline getindex(A::IntegralArray, ids::Tuple...) = getindex(A, first(ids)...)

"""
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
"""
function sum_region(
	integral_image_arr::AbstractArray,
	top_left::Tuple{T, T},
	bottom_right::Tuple{T, T}
) where T <: Integer
    _1 = one(T)
	_0 = zero(0)
	sum = integral_image_arr[last(bottom_right), first(bottom_right)]
    sum -= first(top_left) > _1 ? integral_image_arr[last(bottom_right), first(top_left) - _1] : _0
    sum -= last(top_left) > _1 ? integral_image_arr[last(top_left) - _1, first(bottom_right)] : _0
    sum += last(top_left) > _1 && first(top_left) > _1 ? integral_image_arr[last(top_left) - _1, first(top_left) - _1] : _0
    return sum
end

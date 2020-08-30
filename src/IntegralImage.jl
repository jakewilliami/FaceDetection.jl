#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
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

module IntegralImage

import Base: size, getindex, LinearIndices
using Images: Images, coords_spatial


export toIntegralImage, sumRegion


struct IntegralArray{T, N, A} <: AbstractArray{T, N}
	data::A
end


function toIntegralImage(imgArr::AbstractArray)
	#=
    Calculates the integral image based on this instance's original image data.
    
    parameter `imgArr`: Image source data [type: Abstract Array]
    
    return `integralImageArr`: Integral image for given image [type: Abstract Array]
    
    https://www.ipol.im/pub/art/2014/57/article_lr.pdf, p. 346
    
    This function is adapted from https://github.com/JuliaImages/IntegralArrays.jl/blob/a2aa5bb7c2d26512f562ab98f43497d695b84701/src/IntegralArrays.jl
    =#
	
	arraySize = size(imgArr)
    integralImageArr = Array{Images.accum(eltype(imgArr))}(undef, arraySize)
    sd = coords_spatial(imgArr)
    cumsum!(integralImageArr, imgArr; dims=sd[1])#length(arraySize)
    for i = 2:length(sd)
        cumsum!(integralImageArr, integralImageArr; dims=sd[i])
    end
	
    return Array{eltype(imgArr), ndims(imgArr)}(integralImageArr)
end

LinearIndices(A::IntegralArray) = Base.LinearFast()
size(A::IntegralArray) = size(A.data)
getindex(A::IntegralArray, i::Int...) = A.data[i...]
getindex(A::IntegralArray, ids::Tuple...) = getindex(A, ids[1]...)


function sumRegion(integralImageArr::AbstractArray, topLeft::Tuple{Int64,Int64}, bottomRight::Tuple{Int64,Int64})
    #=
    parameter `integralImageArr`: The intermediate Integral Image [type: Abstract Array]
    Calculates the sum in the rectangle specified by the given tuples:
        parameter `topLeft`: (x,y) of the rectangle's top left corner [type: Tuple]
        parameter `bottomRight`: (x,y) of the rectangle's bottom right corner [type: Tuple]
    
    return: The sum of all pixels in the given rectangle defined by the parameters topLeft and bottomRight
    =#
	
	sum = integralImageArr[bottomRight[2], bottomRight[1]]
    sum -= topLeft[1] > 1 ? integralImageArr[bottomRight[2], topLeft[1] - 1] : 0
    sum -= topLeft[2] > 1 ? integralImageArr[topLeft[2] - 1, bottomRight[1]] : 0
    sum += topLeft[2] > 1 && topLeft[1] > 1 ? integralImageArr[topLeft[2] - 1, topLeft[1] - 1] : 0
	
    return sum
end

end # end module

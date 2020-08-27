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


function toIntegralImage(imgArr::AbstractArray)
    #=
    Calculates the integral image based on this instance's original image data.
    
    parameter `imgArr`: Image source data [type: Abstract Array]
    
    return `integralImageArr`: Integral image for given image [type: Abstract Array]
    
    https://www.ipol.im/pub/art/2014/57/article_lr.pdf, p. 346
    =#
    
    arrRows, arrCols = size(imgArr) # get size only once in case
    rowSum = zeros(arrRows, arrCols)
    integralImageArr = zeros(arrRows, arrCols)
    
    # process the first column
    for x in 1:(arrRows)
        # we cannot access an element that does not exist if we are in the top left corner of the image matrix
        if isone(x)
            integralImageArr[x, 1] = imgArr[x, 1]
        else
            integralImageArr[x, 1] = integralImageArr[x-1, 1] + imgArr[x, 1]
        end
    end
    
    # start processing columns
    for y in 1:(arrCols)
        # same as above: we cannot access a 0th element in the matrix our scalar accumulator s will catch the 1st row, so we only needed to predefine the first column before this loop
        if isone(y)
            continue
        end
        
        # get initial row
        s = imgArr[1, y] # scalar accumulator
        integralImageArr[1, y] = integralImageArr[1, y-1] + s
        
        # now start processing everything else
        for x in 1:(arrRows)
            if isone(x)
                continue
            end
            s = s + imgArr[x, y]
            integralImageArr[x, y] = integralImageArr[x, y-1] + s
        end
    end
    
    return integralImageArr

end
    

function sumRegion(integralImageArr::AbstractArray, topLeft::Tuple{Int64,Int64}, bottomRight::Tuple{Int64,Int64})
    #=
    parameter `integralImageArr`: The intermediate Integral Image [type: Abstract Array]
    Calculates the sum in the rectangle specified by the given tuples:
        parameter `topLeft`: (x,y) of the rectangle's top left corner [type: Tuple]
        parameter `bottomRight`: (x,y) of the rectangle's bottom right corner [type: Tuple]
    
    return: The sum of all pixels in the given rectangle defined by the parameters topLeft and bottomRight
    =#
    
    # swap tuples
    topLeft = (topLeft[2], topLeft[1])
    bottomRight = (bottomRight[2], bottomRight[1])
    
    if isequal(topLeft, bottomRight)
        return integralImageArr[topLeft[1], topLeft[2]]
    end
    
    # construct rectangles
    topRight = (bottomRight[1], topLeft[2])
    bottomLeft = (topLeft[1], bottomRight[2])
    
    topLeftVal = integralImageArr[topLeft[1], topLeft[2]]
    bottomRightVal = integralImageArr[bottomRight[1], bottomRight[2]]
    topRightVal = integralImageArr[topRight[1], topRight[2]]
    bottomLeftVal = integralImageArr[bottomLeft[1], bottomLeft[2]]
    
    return bottomRightVal - topRightVal - bottomLeftVal + topLeftVal
end


export toIntegralImage
export sumRegion

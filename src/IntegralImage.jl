#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
"""
Any one pixel in a given image has the value that is the sum of all of the pixels above it and to the left.
Original    Integral
+--------   +------------
| 1 2 3 .   | 1  3  6 .
| 4 5 6 .   | 5 12 21 .
| . . . .   | . . . . .
"""

function toIntegralImage(imgArr::AbstractArray)
    """
    https://www.ipol.im/pub/art/2014/57/article_lr.pdf, p. 346
    """
    
    arrRows, arrCols = size(imgArr) # get size only once in case
    rowSum = zeros(arrRows, arrCols)
    integralImageArr = zeros(arrRows, arrCols)
    
    # process the first column
    for x in 1:(arrRows)
        # we cannot access an element that does not exist if we are in the
        # top left corner of the image matrix
        if isone(x)
            integralImageArr[x, 1] = imgArr[x, 1]
        else
            integralImageArr[x, 1] = integralImageArr[x-1, 1] + imgArr[x, 1]
        end
    end
    
    # start processing columns
    for y in 1:(arrCols)
        # same as above: we cannot access a 0th element in the matrix
        # our scalar accumulator s will catch the 1st row, so we only
        # needed to predefine the first column before this loop
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
    

function sumRegion(integralImageArr::AbstractArray, topLeft::AbstractArray, bottomRight::AbstractArray)
    """
    Calculates the sum in the rectangle specified by the given tuples
    topLeft: (x, y) of the rectangle's top left corner
    bottomRight: (x, y) of the rectangle's bottom right corner
    return The sum of all pixels in the given rectangle
    """
    # swap tuples
    topLeft = Array(topLeft[2], topLeft[1])
    bottomRight = Array(bottomRight[2], bottomRight[1])
    
    if isequal(topLeft, bottomRight)
            return integralImageArr[topLeft]
    end
    
    # construct rectangles
    topRight = Array(bottomRight[1], topLeft[2])
    bottomLeft = Array(topLeft[1], bottomRight[2])
    
    return integralImageArr[bottomRight] - integralImageArr[topLeft] - integralImageArr[bottomLeft] + integralImageArr[topLeft]
end



# imgArr = getImageMatrix()
# integralImageArr = toIntegralImage(imgArr)

# Base.print_matrix(stdout, integralImageArr)
# println(size(integralImageArr))

export toIntegralImage
export sumRegion

# return integralImageArr




### TESTING

# A = [1 7 4 2 9;7 2 3 8 2;1 8 7 9 1;3 2 3 1 5;2 9 5 6 6]
#
# Base.print_matrix(stdout, A)
# println(size(A))
#
# Base.print_matrix(stdout, toIntegralImage(A))
# println(size(toIntegralImage(A)))

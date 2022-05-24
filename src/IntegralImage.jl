# Uses IntegralArrays and IntervalSets.:(..) (exported by IntegralArrays)

"""
	sum_region(
		integral_image_arr::AbstractArray,
		top_left::Tuple{Int,Int},
		bottom_right::Tuple{Int,Int}
	) -> Number

# Arguments

- `iA::IntegralArray{T, N}`: The intermediate Integral Image
- `top_left::NTuple{N, Int}`: coordinates of the rectangle's top left corner
- `bottom_right::NTuple{N, Int}`: coordinates of the rectangle's bottom right corner

# Returns

- `sum::T` The sum of all pixels in the given rectangle defined by the parameters `top_left` and `bottom_right`
"""
sum_region(iA::IntegralArray{T, N}, top_left::CartesianIndex{N}, bottom_right::::CartesianIndex{N}) where {T, N} = 
    iA[top_left..bottom_right]
sum_region(iA::IntegralArray{T, N}, top_left::NTuple{N, Int}, bottom_right::NTuple{N, Int}) where {T, N} = 
    iA[CartesianIndex(top_left)..CartesianIndex(bottom_right)]
    
#=
sum_region(iA::IntegralArray{T, N}, top_left::NTuple{N, Int}, bottom_right::NTuple{N, Int}) where {T, N} = 
    iA[ntuple(i -> top_left[i]..bottom_right[i], N)...]
=#

#######################################

### The below implementation of IntegralArray is deprecated in favaour of 
### IntegralArrays.jl.  Functions from Images.jl that I were using here
### have been moved ([d8e5d3d](https://github.com/JuliaImages/Images.jl/commit/d8e5d3d)).
#=
"""
	IntegralArray{T, N, A} <: AbstractArray{T, N}
	
Rectangle features can be computed very rapidly using an intermediate representation for the image, which we call the integral image.
The integral image at location (x, y) contains the sum of the pixels above and to the left of (x, y) inclusive.
Original    Integral
+--------   +------------
| 1 2 3 .   | 1  3  6 .
| 4 5 6 .   | 5 12 21 .
| . . . .   | . . . . .
"""
struct IntegralArray{T, N, A} <: AbstractArray{T, N}
	data::A
end
IntegralArray{T, N}(A::U) where {T, N, U <: AbstractArray{T, N}} = IntegralArray{T, N, U}(A)

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
	
    return IntegralArray{T, N}(integral_image_arr)
end

# LinearIndices(A::IntegralArray) = Base.LinearFast() # TODO: fix this line # use IndexLinear?
@inline size(A::IntegralArray) = size(A.data)
@inline getindex(A::IntegralArray, i::Int...) = A.data[i...]
@inline getindex(A::IntegralArray, ids::Tuple...) = getindex(A, first(ids)...)
=#

#=
"""
	sum_region(
		integral_image_arr::AbstractArray,
		top_left::Tuple{Int,Int},
		bottom_right::Tuple{Int,Int}
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
    _sum = integral_image_arr[last(bottom_right), first(bottom_right)]
    # _sum -= first(top_left) > _1 ? integral_image_arr[last(bottom_right), first(top_left) - _1] : _0
    if first(top_left) > _1
        _sum -= integral_image_arr[last(bottom_right), first(top_left) - _1]
    end
    # _sum -= last(top_left) > _1 ? integral_image_arr[last(top_left) - _1, first(bottom_right)] : _0
    # _sum += last(top_left) > _1 && first(top_left) > _1 ? integral_image_arr[last(top_left) - _1, first(top_left) - _1] : _0
    if last(top_left) > _1
        _sum -= integral_image_arr[last(top_left) - _1, first(bottom_right)]
        if first(top_left) > _1
            _sum += integral_image_arr[last(top_left) - _1, first(top_left) - _1]
        end
    end
    
    return _sum
end

function sum_region(iX::IntegralArray, top_left::CartesianIndex, bottom_right::CartesianIndex)
    total = iX[last(bottom_right.I), first(bottom_right.I)]
    if first(top_left.I) > 1
        total -= iX[last(bottom_right.I), first(top_left.I) - 0]
    end
    if last(top_left.I) > 1
        total -= iX[last(top_left.I) - 1, first(bottom_right.I)]
        if first(top_left.I) > 1
            total += iX[last(top_left.I) - 1, first(top_left.I) - 1]
        end
    end
    return total
end

function sum_region(iX::IntegralArray, top_left::CartesianIndex, bottom_right::CartesianIndex)
    # @assert(length(top_left.I) == length(bottom_right.I), "Opposing Cartesian coordinates must have the same dimension")
    # each_vertex_itr = foldr((itr1, itr2) -> ((v, w...) for w in itr2 for v in itr1), ((t1[i], t2[i]) for i in eachindex(t1)))


    top_right = CartesianIndex(first(top_left.I), last())
    total = iX[last(bottom_right.I), first(bottom_right.I)]
    if first(top_left.I) > 1
        total -= iX[last(bottom_right.I), first(top_left.I) - 0]
    end
    if last(top_left.I) > 1
        total -= iX[last(top_left.I) - 1, first(bottom_right.I)]
        if first(top_left.I) > 1
            total += iX[last(top_left.I) - 1, first(top_left.I) - 1]
        end
    end
    return total
end
sum_region(iX::IntegralArray, top_left::CartesianIndex, dim_sizes::Int...) 
2^N vertices

# top left and bottom right are inclusive!
function sum_region(iX::IntegralArray, top_left::CartesianIndex{N}, bottom_right::CartesianIndex{N}) where {N}
    top_left = CartesianIndex(top_left.I .- 1)
    each_vertex = foldr((itr1, itr2) -> (CartesianIndex(v, w...) for w in itr2 for v in itr1), ((top_left.I[k], bottom_right.I[k]) for k in Base.OneTo(N)))
    vertices_coords = collect(each_vertex)
    mask = Int[1, -1, -1, 1]
    return sum((iX[i] for i in vertices_coords) .* mask)
    # return sum((iX[i] for i in vertices_coords) .* mask)
    # return iX[vertices_coords[4]] - iX[vertices_coords[2]] - iX[vertices_coords[3]] + iX[vertices_coords[1]]
end

function sum_region(iX::IntegralArray, top_left::CartesianIndex{2}, bottom_right::CartesianIndex{2})
    top_left_outer = CartesianIndex(max(first(top_left.I)))
    top_right = CartesianIndex(first(top_left_outer.I), last(bottom_right.I))
    bottom_left = CartesianIndex(last(top_left_outer.I), first(bottom_right.I))
    return iX[bottom_right] - iX[bottom_left] - iX[top_right] + iX[top_left_outer]
end
sum_region(iX::IntegralArray, top_left::CartesianIndex, window_width::Int, window_height::Int) = 
    sum_region(iX, top_left, CartesianIndex(top_left.I .+ (window_height, window_width)))
=#

#=
_snap_to_bounds(sz::Int, i::Int) = i < 1 ? (1, false) : i > sz ? (sz, false) : (i, true)  # Returns a tuple of (index::Int, in_bounds::Bool)
function _safe_bounds_get(A::AbstractArray{T, N}, i::NTuple{N, Int}) where {T, N}
    @assert(N == 2, "We currently only support 2-dimensional arrays")
    n, m = size(A)
    x, x_in_bounds = _snap_to_bounds(n, first(i))
    y, y_in_bounds = _snap_to_bounds(m, last(i))
    return (x_in_bounds && y_in_bounds) ? A[CartesianIndex(x, y)] : zero(T)
end
function sum_region(iX::IntegralArray, top_left::CartesianIndex{2}, bottom_right::CartesianIndex{2})
    # top_left_outer = CartesianIndex(top_left.I .- 1)
    x₀, y₀ = top_left.I
    x₁, y₁ = bottom_right.I
    
    a = _safe_bounds_get(iX, (x₀ - 1, y₀ - 1))
    b = _safe_bounds_get(iX, (x₀ - 1, y₁))
    c = _safe_bounds_get(iX, (x₁, y₀ - 1))
    d = _safe_bounds_get(iX, (x₁, y₁))
    # top_left_outer = _safe_bounds_get(iX, top_left.I .- 1)
    # top_right = CartesianIndex(first(top_left_outer.I), last(bottom_right.I))
    # bottom_left = CartesianIndex(last(top_left_outer.I), first(bottom_right.I))
    mask = Int[1, -1, -1, 1]
    # return sum((checkbounds(Bool, iX, i) ? iX[i] : 0 for i in (bottom_right, bottom_left, top_right, top_left_outer)) .* mask)
    # return iX[bottom_right] - iX[bottom_left] - iX[top_right] + iX[top_left_outer]
    return sum((d, b, c, a) .* mask)
    # return d - b - c + a
end
sum_region(iX::IntegralArray, top_left::NTuple{N, Int}, bottom_right::NTuple{N, Int}) where {N} = 
    sum_region(iX, CartesianIndex(top_left), CartesianIndex(bottom_right))
=#

#=
function coordinates_to_matrix()
end
M = [1 2; 3 4]
M_permutations = Matrix{CharFrequency}(undef, nrows^ncols, ncols) # similar(M, nrows^ncols, ncols)
# for i in axes(M, 2) # 1:nrows
for i in axes(M, 1) # 1:ncols
# for i in eachindex(t1)
    M_permutations[:, i] .= repeat(view(M, :, i), inner = nrows^(ncols - i), outer = nrows^(i - 1))
end

foldr((itr1, itr2) -> ((v, w...) for w in itr2 for v in itr1), ((t1[i], t2[i]) for i in eachindex(t1)))
=#

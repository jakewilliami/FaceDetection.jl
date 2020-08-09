#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/src/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#

#Adapted from IndirectArray

module OpenCVImages

dtypes = Union{UInt8, Int8, UInt16, Int16, Int32, Float32, Float64}

struct OpenCVImage{T <: dtypes} <: AbstractArray{T,3}
    mat
    data_raw
    data

    @inline function OpenCVImage{T}(mat, data_raw::AbstractArray{T,3}) where {T <: dtypes}
        data = reinterpret(T, data_raw)
        new{T}(mat, data_raw, data)
    end

    @inline function OpenCVImage{T}(data_raw::AbstractArray{T, 3}) where {T <: dtypes}
        data = reinterpret(T, data_raw)
        mat = nothing
        new{T}(mat, data_raw, data)
    end
end

function Base.deepcopy_internal(x::OpenCVImage{T}, y::IdDict) where {T}
    if haskey(y, x)
        return y[x]
    end
    ret = Base.copy(x)
    y[x] = ret
    return ret
end

Base.size(A::OpenCVImage) = size(A.data)
Base.axes(A::OpenCVImage) = axes(A.data)
Base.IndexStyle(::Type{OpenCVImage{T}}) where {T} = IndexCartesian()

Base.strides(A::OpenCVImage{T}) where {T} = strides(A.data)
Base.copy(A::OpenCVImage{T}) where {T} = OpenCVImage{T}(copy(A.data_raw))
Base.pointer(A::OpenCVImage) = Base.pointer(A.data)

Base.unsafe_convert(::Type{Ptr{T}}, A::OpenCVImage{S}) where {T, S} = Base.unsafe_convert(Ptr{T}, A.data)

@inline function Base.getindex(A::OpenCVImage{T}, I::Vararg{Int,3}) where {T}
    @boundscheck checkbounds(A.data, I...)
    @inbounds ret = A.data[I...]
    ret
end

@inline function Base.setindex!(A::OpenCVImage, x, I::Vararg{Int,3})
    @boundscheck checkbounds(A.data, I...)
    A.data[I...] = x
    return A
end

end # module
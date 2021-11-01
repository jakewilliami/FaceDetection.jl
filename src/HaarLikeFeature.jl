const FEATURE_TYPES = (
    two_vertical = (1, 2),
    two_horizontal = (2, 1),
    three_horizontal = (3, 1),
    three_vertical = (1, 3),
    four = (2, 2)
    )

abstract type AbstractHaarFeature end

"""
    mutable struct HaarLikeObject{I <: Integer, F <: AbstractFloat}

        Struct representing a Haar-like feature.
        
    feature_type::Tuple{I, I}
    position::Tuple{I, I}
    top_left::Tuple{I, I}
    bottom_right::Tuple{I, I}
    width::I
    height::I
    threshold::I
    polarity::I
    weight::F
"""
mutable struct HaarLikeObject{I <: Integer, F <: AbstractFloat} <: AbstractHaarFeature
    #parametric struct to store the ints and floats efficiently
    feature_type::Tuple{I, I}
    position::Tuple{I, I}
    top_left::Tuple{I, I}
    bottom_right::Tuple{I, I}
    width::I
    height::I
    threshold::I
    polarity::I
    weight::F
end # end structure

"""
    HaarLikeObject(
        feature_type::Tuple{Integer, Integer},
        position::Tuple{Integer, Integer},
        width::Integer,
        height::Integer,
        threshold::Integer,
        polarity::Integer
    ) -> HaarLikeObject
"""
function HaarLikeObject(
    feature_type::Tuple{Integer, Integer},
    position::Tuple{Integer, Integer},
    width::Integer,
    height::Integer,
    threshold::Integer,
    polarity::Integer
)

    # make sure that everything is of the same size
    p₁, p₂ = position
    f₁, f₂ = feature_type
    p₁, p₂, f₁, f₂, width, height, threshold, polarity =
        promote(p₁, p₂, f₁, f₂, width, height, threshold, polarity)
    position = (p₁, p₂)
    feature_type = (f₁, f₂)
    top_left = position
    bottom_right = (first(position) + width, last(position) + height)
    weight = float(one(p₁)) #to make a float of the same size
    

    HaarLikeObject(feature_type, position, top_left, bottom_right, width, height, threshold, polarity, weight)
end

"""
    get_score(feature::HaarLikeObject, int_img::AbstractArray) -> Tuple{Number, Number}
    
Get score for given integral image array.  This is the feature cascade.

# Arguments

- `feature::HaarLikeObject`: given Haar-like feature (parameterised replacement of Python's `self`)
- `int_img::AbstractArray`: Integral image array

# Returns

- `score::Number`: Score for given feature
"""
function get_score(feature::HaarLikeObject{I, F}, int_img::AbstractArray{T, N}) where {I, F, T, N}
    score = zero(I)
    faceness = zero(I)
    _2f = F(2)
    _3f = F(3)
    _half = F(0.5)
    _one_third = F(1.0 / 3.0)
    
    if feature.feature_type == FEATURE_TYPES.two_vertical
        _first = sum_region(int_img, feature.top_left, (first(feature.top_left) + feature.width, round(I, last(feature.top_left) + feature.height / 2)))
        second = sum_region(int_img, (first(feature.top_left), round(I, last(feature.top_left) + feature.height / 2)), feature.bottom_right)
        score = _first - second
        faceness = I(1)
    elseif feature.feature_type == FEATURE_TYPES.two_horizontal
        _first = sum_region(int_img, feature.top_left, (round(I, first(feature.top_left) + feature.width / 2), last(feature.top_left) + feature.height))
        second = sum_region(int_img, (round(I, first(feature.top_left) + feature.width / 2), last(feature.top_left)), feature.bottom_right)
        score = _first - second
        faceness = I(2)
    elseif feature.feature_type == FEATURE_TYPES.three_horizontal
        _first = sum_region(int_img, feature.top_left, (round(I, first(feature.top_left) + feature.width / 3), last(feature.top_left) + feature.height))
        second = sum_region(int_img, (round(I, first(feature.top_left) + feature.width / 3), last(feature.top_left)), (round(I, first(feature.top_left) + 2 * feature.width / 3), last(feature.top_left) + feature.height))
        third = sum_region(int_img, (round(I, first(feature.top_left) + 2 * feature.width / 3), last(feature.top_left)), feature.bottom_right)
        score = _first - second + third
        faceness = I(3)
    elseif feature.feature_type == FEATURE_TYPES.three_vertical
        _first = sum_region(int_img, feature.top_left, (first(feature.bottom_right), round(I, last(feature.top_left) + feature.height / 3)))
        second = sum_region(int_img, (first(feature.top_left), round(I, last(feature.top_left) + feature.height / 3)), (first(feature.bottom_right), round(I, last(feature.top_left) + 2 * feature.height / 3)))
        third = sum_region(int_img, (first(feature.top_left), round(I, last(feature.top_left) + 2 * feature.height / 3)), feature.bottom_right)
        score = _first - second + third
        faceness  = I(4)
    elseif feature.feature_type == FEATURE_TYPES.four
        # top left area
        _first = sum_region(int_img, feature.top_left, (round(I, first(feature.top_left) + feature.width / 2), round(I, last(feature.top_left) + feature.height / 2)))
        # top right area
        second = sum_region(int_img, (round(I, first(feature.top_left) + feature.width / 2), last(feature.top_left)), (first(feature.bottom_right), round(I, last(feature.top_left) + feature.height / 2)))
        # bottom left area
        third = sum_region(int_img, (first(feature.top_left), round(I, last(feature.top_left) + feature.height / 2)), (round(I, first(feature.top_left) + feature.width / 2), last(feature.bottom_right)))
        # bottom right area
        fourth = sum_region(int_img, (round(I, first(feature.top_left) + feature.width / 2), round(I, last(feature.top_left) + feature.height / 2)), feature.bottom_right)
        score = _first - second - third + fourth
        faceness = I(5)
    end
    
    return score, faceness
end

"""
    get_vote(feature::HaarLikeObject, int_img::AbstractArray) -> Integer

Get vote of this feature for given integral image.

# Arguments

- `feature::HaarLikeObject`: given Haar-like feature (parameterised replacement of Python's `self`)
- `int_img::AbstractArray`: Integral image array [type: Abstract Array]

# Returns

 - `vote::Integer`:
    1       ⟺ this feature votes positively
    -1      otherwise
"""
function get_vote(feature::HaarLikeObject{I, F}, int_img::AbstractArray{T, N}) where {I, F, T, N}
    score, _ = get_score(feature, int_img) # we only care about score here, not faceness
    # return (feature.weight * score) < (feature.polarity * feature.threshold) ? one(Int8) : -one(Int8)
    # return feature.weight * (score < feature.polarity * feature.threshold ? one(Int8) : -one(Int8))
    return score < feature.polarity * feature.threshold ? feature.weight : -feature.weight
    # self.weight * (1 if score < self.polarity * self.threshold else -1)
end

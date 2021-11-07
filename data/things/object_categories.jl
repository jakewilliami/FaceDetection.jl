# Return a list of object categories from the images
function get_object_categories(object_images::Vector{String})
    object_categories = String[]
    for object_image in object_images
        object_category = join(split(object_image, '_')[1:end-1], ' ')
        if object_category ∉ object_categories
            push!(object_categories, object_category)
        end
    end
    return object_categories
end
get_object_categories(object_image_dir::String) = 
    get_object_categories(readdir(object_image_dir))

# Filter out animals from the categories
function filter_out_animals(object_image_categories::Vector{String})
    animals = readlines(download("https://gist.githubusercontent.com/atduskgreg/3cf8ef48cb0d29cf151bedad81553a54/raw/82f142562cf50b0f6fb8010f890b2f934093553e/animals.txt"))
    animals = String[string(lowercase(animal)) for animal in animals]
    filtered_categories = String[]
    for image_category in object_image_categories
        if image_category ∉ animals || !any(startswith(image_category, animal) for animal in animals)
            push!(filtered_categories, image_category)
        else
            @info "$image_category is being filtered out"
        end
    end
    return filtered_categories
end
filter_out_animals(object_image_dir::String) = 
    filter_out_animals(get_object_categories(object_image_dir))

# Get the category lists and write them to file
function main()
    all_categories = get_object_categories(joinpath(@__DIR__, "object_images"))
    all_categories_minus_animals = filter_out_animals(all_categories)
    
    open("all_categories.txt", "w") do io
        for category in all_categories
            write(io, category, '\n')
        end
    end
    
    open("all_categories_animals_filtered.txt", "w") do io
        for category in all_categories_minus_animals
            write(io, category, '\n')
        end
    end
    
    return ("all_categories.txt", "all_categories_animals_filtered.txt")
end

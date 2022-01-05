get_category_from_image_name(s::String) = join(split(basename(s), '_')[1:(end - 1)], ' ')

# Return a list of object categories from the images
function get_object_categories(object_images::Vector{String})
    object_categories = String[]
    for object_image in object_images
        object_image = basename(object_image)
        object_category = get_category_from_image_name(object_image)
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
        category_is_animal = image_category ∈ animals
        # category_starts_with_animal = any(startswith(image_category, animal) for animal in animals)
        if !category_is_animal # || !category_starts_with_animal
            push!(filtered_categories, image_category)
        end
    end
    return filtered_categories
end
filter_out_animals(object_image_dir::String) = 
    filter_out_animals(get_object_categories(object_image_dir))

# Get the category lists and write them to file
function main(all_object_image_dir::String)
    outfile_all_categories_list = "all_categories.txt"
    outfile_all_categories_filtered_list = "all_categories_filtered.txt"
    misc_filter_categories_list = "misc_filter_categories.txt"
    
    all_object_images = readdir(all_object_image_dir, sort = true, join = true)
    
    all_categories = get_object_categories(all_object_images)
    all_categories_filtered = filter_out_animals(all_categories)
    misc_filter_categories = readlines(misc_filter_categories_list)
    filter!(category -> category ∉ misc_filter_categories, all_categories_filtered)
    
    open(outfile_all_categories_list, "w") do io
        for category in all_categories
            write(io, category, '\n')
        end
    end
    
    open(outfile_all_categories_filtered_list, "w") do io
        for category in all_categories_filtered
            write(io, category, '\n')
        end
    end
    
    @info "There are currently $(length(all_object_images)) images in your object directory"
    categories_warned = String[]
    removed = 0
    for object_image in all_object_images
        object_category = get_category_from_image_name(object_image)
        if object_category ∉ all_categories_filtered
            if object_category ∉ categories_warned
                @warn("Removing images of the category \"$object_category\"")
                push!(categories_warned, object_category)
            end
	    @info "rm $object_image"
            try rm(object_image) catch e @warn(e) end
            removed += 1
        end
    end
    @info "We have removed all of the images that needed removing, and are left with $(length(all_object_images) - removed) images in your object directory"
    
    return nothing
end

main("object_images/")

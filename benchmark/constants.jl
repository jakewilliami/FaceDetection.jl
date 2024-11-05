# Use same data as tests, for simplicity
main_data_path = joinpath(dirname(@__DIR__), "test", "images")

pos_training_path = joinpath(main_data_path, "pos")
neg_training_path = joinpath(main_data_path, "neg")
pos_testing_path = joinpath(main_data_path, "pos_testing")
neg_testing_path = joinpath(main_data_path, "neg_testing")

num_classifiers = 10
min_size_img = (19, 19)
scale, scale_to = false, (200, 200)

min_feature_height = 8
max_feature_height = 10
min_feature_width = 8
max_feature_width = 10

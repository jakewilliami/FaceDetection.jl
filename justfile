# -*- mode: just -*-

# Define project directory
#
# {Project, Manifest}.toml should exist in the root directory, as per
# the justfile, so we should be able to use the justfile directory [1]
# as the root project directory that Julia uses.
#
# [1]: https://just.systems/man/en/functions.html#justfile-and-justfile-directory
project_dir := justfile_dir() / "/"
test_dir := project_dir / "test/"
test_file := test_dir / "runtests.jl"
docs_dir := project_dir / "docs/"
docs_mk_file := docs_dir / "make.jl"
bench_dir := project_dir / "benchmark/"
bench_file := bench_dir / "basic.jl"
standard_instantiate_code := """
import Pkg
Pkg.instantiate()
"""
dev_instantiate_code := """
import Pkg
Pkg.develop(Pkg.PackageSpec(path=pwd()))
Pkg.instantiate()
"""

# Test project
test:
    julia --project={{project_dir}} {{test_file}}

# Run specified file
run run_file:
    julia --project={{project_dir}} {{run_file}}

# Generate documentation
[group: 'ci']
docs: (instantiate-dev docs_dir)
    julia --project={{docs_dir}} {{docs_mk_file}}

# Benchmark performance
[group: 'ci']
bench: (instantiate-dev bench_dir)
    julia --project={{bench_dir}} {{bench_file}}

# Check formatting with blue style
[group: 'ci']
fmt: install-formatter
    # https://github.com/invenia/BlueStyle
    julia --project=@JuliaFormatter -e 'using JuliaFormatter; format("{{project_dir}}", style=BlueStyle())'

# Install JuliaFormatter
[private]
install-formatter:
    julia --project=@JuliaFormatter -e 'import Pkg; Pkg.add("JuliaFormatter")'

# Instantiate main project
instantiate:
    julia --project={{project_dir}} -e '{{standard_instantiate_code}}'

# Instantiate sub-project
[private]
instantiate-dev dev_project_dir:
    julia --project={{dev_project_dir}} -e '{{dev_instantiate_code}}'


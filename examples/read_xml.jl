#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#

println("\033[1;34m===>\033[0;38m\033[1;38m\tLoading required libraries (it will take a moment to precompile if it is your first time doing this)...\033[0;38m")

include(joinpath(dirname(dirname(@__FILE__)), "src", "FaceDetection.jl"))

using .FaceDetection
const FD = FaceDetection
using Images: imresize
using StatsPlots  # StatsPlots required for box plots # plot boxplot @layout :origin savefig
using CSV: write
using DataFrames: DataFrame
using HypothesisTests: UnequalVarianceTTest
using LightXML

println("...done")

function main()
    xml_data = parse_file(joinpath(dirname(dirname(@__FILE__)), "data", "haarcascades", "haarcascade_frontalface_default.xml")) # ../data/haarcascades/haarcascade_frontalface_default.xml

	for c in child_nodes(root(xml_data))  # c is an instance of XMLNode
    if is_elementnode(c)
        e = XMLElement(c)  # this makes an XMLElement instance
        println(name(e))
    end
end
end

main()

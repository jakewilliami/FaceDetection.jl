#! /usr/bin/env bash

trap "exit" INT

if [[ $(uname -s) != "Darwin" ]]
then
	echo "OpenCV has been ported to Julia, but has only been tested on OSX.  Please keep this in mind when using this implementation of Viola-Jone."
fi

echo "Please ensure FaceDetection.jl is installed in your home directory..."
sleep 5

if [[ -z $(command -v julia) ]]
then
	echo "You do not have Julia installed.  Please rectify this.  Ensure you are using version 0.5 or later.  Version 1.5 is preferrable."
fi

echo "Downloading dependencies"
julia -E 'import Pkg; Pkg.activate(joinpath(homedir(), "FaceDetection.jl")); Pkg.instantiate()'

brew list > /tmp/brewlist
if ! grep "^opencv@3$" /tmp/brewlist > /dev/null 2>&1
then
	echo "Downloading OpenCV"
	brew install opencv@3
fi

echo "Downloading Cxx.jl"
julia -E 'cd(joinpath(homedir(), "FaceDetection.jl")); import Pkg; Pkg.add(url="https://github.com/Keno/Cxx.jl"); Pkg.build("Cxx")'

echo "Installing OpenCV packages for Julia"
julia -E 'cd(joinpath(homedir(), "FaceDetection.jl")); import Pkg; Pkg.add(url="https://github.com/JuliaOpenCV/CVCore.jl"); Pkg.add(url="https://github.com/JuliaOpenCV/CVCalib3d.jl"); Pkg.add(url="https://github.com/JuliaOpenCV/CVHighGUI.jl"); Pkg.add(url="https://github.com/JuliaOpenCV/CVVideoIO.jl"); Pkg.add(url="https://github.com/JuliaOpenCV/CVImgProc.jl"); Pkg.add(url="https://github.com/JuliaOpenCV/CVImgCodecs.jl"); Pkg.add(url="https://github.com/JuliaOpenCV/LibOpenCV.jl"); Pkg.add(url="https://github.com/JuliaOpenCV/OpenCV.jl"); Pkg.build("LibOpenCV"); Pkg.test("OpenCV")'

# Benchmarking Results

These are results from benchmarking the training process.  The following are benchmarking results from running equivalent programmes in both repositories.  These programmes uses ~10 thousand training images at 19 x 19 pixels each.

Language of Implementation | Commit | Run Time in Seconds | Number of Allocations | Memory Usage
--- | --- | --- | --- | ---
[Python](https://github.com/Simon-Hohberg/Viola-Jones/) | [8772a28](https://github.com/Simon-Hohberg/Viola-Jones/commit/8772a28) | 480.0354 | —ᵃ | —ᵃ
[Julia](https://github.com/jakewilliami/FaceDetection.jl/) | [6fd8ca9e](https://github.com/jakewilliami/FaceDetection.jl/commit/6fd8ca9e) |19.9057 | 255600105 | 5.11 GiB

ᵃI have not yet figured out benchmarking memory usage in Python.

These results were run on this machine:
```julia
julia> versioninfo()
Julia Version 1.5.2
Commit 539f3ce943 (2020-09-23 23:17 UTC)
Platform Info:
  OS: macOS (x86_64-apple-darwin18.7.0)
  CPU: Intel(R) Core(TM) i5-6360U CPU @ 2.00GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-9.0.1 (ORCJIT, skylake)
```

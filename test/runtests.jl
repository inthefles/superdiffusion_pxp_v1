# runtests.jl
# Test runner for PXPTransport package

using Test
using PXPTransport

@testset "PXPTransport Tests" begin
    include("test_hamiltonian.jl")
    include("test_ed_benchmark.jl")
end

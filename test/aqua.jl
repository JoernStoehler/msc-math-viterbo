using Test
using Aqua
using ViterboConjecture

@testset "Aqua quality" begin
    Aqua.test_all(ViterboConjecture)
end


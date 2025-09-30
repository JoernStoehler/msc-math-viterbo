using Test
using ViterboConjecture

@testset "ViterboConjecture" begin
    @test hello_area_estimate(0) == 0
    @test hello_area_estimate(3) == 9
    @test hello_area_estimate(-2) == 4
end


using TensorFunctions
using Test

@testset "TensorFunctions.jl" begin
    @test TensorFunctions.isinpairedindex(:( :a ),:lhs) == true
    @test TensorFunctions.isinpairedindex(:( :b|1 ),:lhs) == false
    @test TensorFunctions.isinpairedindex(:( :b|size(foo,1) ),:lhs) == false
    @test TensorFunctions.isinpairedindex(:( :a ),:rhs) == true
    @test TensorFunctions.isinpairedindex(:( :b|1 ),:rhs) == true
    @test TensorFunctions.isinpairedindex(:( :b|size(foo,1) ),:rhs) == true

    @test TensorFunctions.ispairedindex(:( :a ),:lhs) == false
    @test TensorFunctions.ispairedindex(:( (:a,:b) ),:lhs) == true
    @test TensorFunctions.ispairedindex(:( (:a,:b|1) ),:lhs) == false
    @test TensorFunctions.ispairedindex(:( (:a|foo(bar),:b) ),:lhs) == false
    @test TensorFunctions.ispairedindex(:( :a ),:rhs) == false
    @test TensorFunctions.ispairedindex(:( (:a,:b) ),:rhs) == true
    @test TensorFunctions.ispairedindex(:( (:a,:b|1) ),:rhs) == true
    @test TensorFunctions.ispairedindex(:( (:a|foo(bar),:b) ),:rhs) == true
end

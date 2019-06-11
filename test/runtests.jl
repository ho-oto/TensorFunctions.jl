using TensorFunctions
using Test

@testset "TensorFunctions.jl" begin
    @test TensorFunctions.issymbol(:( :b )) == true
    @test TensorFunctions.issymbol(:( b )) == false

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

    @test TensorFunctions.isindexproduct(:( :a*5 )) == true
    @test TensorFunctions.isindexproduct(:( :a*hoge() )) == false
    @test TensorFunctions.isindexproduct(:( :a )) == false
    @test TensorFunctions.isindexproduct(:( :a*5*5 )) == false

    @test TensorFunctions.istensor(:( foo[:a,:b,:c] ),:rhs) == true
    @test TensorFunctions.istensor(:( foo(bar)[:a,:b,:c] ),:rhs) == true
    @test TensorFunctions.istensor(:( foo(bar)[(:a,:b),:c] ),:rhs) == true
    @test TensorFunctions.istensor(:( foo(bar)[(:a|hoge(2),:b),:c] ),:rhs) == true
    @test TensorFunctions.istensor(:( foo(bar|>hugo)[(:a|4,:b|hoge(3)),:c*6] ),:rhs) == true
    @test TensorFunctions.istensor(:( [:a,:b,:c] ),:rhs) == false
    @test TensorFunctions.istensor(:( [:a,(:b,:c)] ),:rhs) == false
    @test TensorFunctions.istensor(:( [(:a|4,:b|hoge(3)),:c*6] ),:rhs) == false

    @test TensorFunctions.istensor(:( foo[:a,:b,:c] ),:lhs) == true
    @test TensorFunctions.istensor(:( foo(bar)[:a,:b,:c] ),:lhs) == true
    @test TensorFunctions.istensor(:( foo(bar)[(:a,:b),:c] ),:lhs) == true
    @test TensorFunctions.istensor(:( foo(bar)[(:a|hoge(2),:b),:c] ),:lhs) == false
    @test TensorFunctions.istensor(:( foo(bar|>hugo)[(:a|4,:b|hoge(3)),:c*6] ),:lhs) == false
    @test TensorFunctions.istensor(:( [:a,:b,:c] ),:lhs) == true
    @test TensorFunctions.istensor(:( [:a,(:b,:c)] ),:lhs) == true
    @test TensorFunctions.istensor(:( [(:a|4,:b|hoge(3)),:c*6] ),:lhs) == false

    @test TensorFunctions.issimpletensor(:( foo[:a,:b,:c] )) == true
    @test TensorFunctions.issimpletensor(:( foo(bar)[:a,:b,:c] )) == true
    @test TensorFunctions.issimpletensor(:( foo(bar)[(:a,:b),:c] )) == false
    @test TensorFunctions.issimpletensor(:( foo(bar)[(:a|hoge(2),:b),:c] )) == false
    @test TensorFunctions.issimpletensor(:( foo(bar|>hugo)[(:a|4,:b|hoge(3)),:c*6] )) == false
    @test TensorFunctions.issimpletensor(:( [:a,:b,:c] )) == false
    @test TensorFunctions.issimpletensor(:( [:a,(:b,:c)] )) == false
    @test TensorFunctions.issimpletensor(:( [(:a|4,:b|hoge(3)),:c*6] )) == false

end

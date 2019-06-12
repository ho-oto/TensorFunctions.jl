using TensorFunctions
using Test

@testset "TensorFunctions.jl" begin
    @test TensorFunctions.issymbol(:( :b )) == true
    @test TensorFunctions.issymbol(:( b )) == false

    @test TensorFunctions.isinpairedindex(:( :a ),false) == true
    @test TensorFunctions.isinpairedindex(:( :b|1 ),false) == false
    @test TensorFunctions.isinpairedindex(:( :b|size(foo,1) ),false) == false

    @test TensorFunctions.isinpairedindex(:( :a )) == true
    @test TensorFunctions.isinpairedindex(:( :b|1 )) == true
    @test TensorFunctions.isinpairedindex(:( :b|size(foo,1) )) == true

    @test TensorFunctions.ispairedindex(:( :a ),false) == false
    @test TensorFunctions.ispairedindex(:( (:a,:b) ),false) == true
    @test TensorFunctions.ispairedindex(:( (:a,:b|1) ),false) == false
    @test TensorFunctions.ispairedindex(:( (:a|foo(bar),:b) ),false) == false

    @test TensorFunctions.ispairedindex(:( :a )) == false
    @test TensorFunctions.ispairedindex(:( (:a,:b) )) == true
    @test TensorFunctions.ispairedindex(:( (:a,:b|1) )) == true
    @test TensorFunctions.ispairedindex(:( (:a|foo(bar),:b) )) == true

    @test TensorFunctions.isindexproduct(:( :a*5 )) == true
    @test TensorFunctions.isindexproduct(:( :a*hoge() )) == false
    @test TensorFunctions.isindexproduct(:( :a )) == false
    @test TensorFunctions.isindexproduct(:( :a*5*5 )) == false

    @test TensorFunctions.istensor(:( foo[:a,:b,:c] )) == true
    @test TensorFunctions.istensor(:( foo(bar)[:a,:b,:c] )) == true
    @test TensorFunctions.istensor(:( foo(bar)[(:a,:b),:c] )) == true
    @test TensorFunctions.istensor(:( foo(bar)[(:a|hoge(2),:b),:c] )) == true
    @test TensorFunctions.istensor(:( foo(bar|>hugo)[(:a|4,:b|hoge(3)),:c*6] )) == true
    @test TensorFunctions.istensor(:( [:a,:b,:c] )) == false
    @test TensorFunctions.istensor(:( [:a,(:b,:c)] )) == false
    @test TensorFunctions.istensor(:( [(:a|4,:b|hoge(3)),:c*6] )) == false

    @test TensorFunctions.istensor(:( foo[:a,:b,:c] ),false) == true
    @test TensorFunctions.istensor(:( foo(bar)[:a,:b,:c] ),false) == true
    @test TensorFunctions.istensor(:( foo(bar)[(:a,:b),:c] ),false) == true
    @test TensorFunctions.istensor(:( foo(bar)[(:a|hoge(2),:b),:c] ),false) == false
    @test TensorFunctions.istensor(:( foo(bar|>hugo)[(:a|4,:b|hoge(3)),:c*6] ),false) == false
    @test TensorFunctions.istensor(:( [:a,:b,:c] ),false) == true
    @test TensorFunctions.istensor(:( [:a,(:b,:c)] ),false) == true
    @test TensorFunctions.istensor(:( [(:a|4,:b|hoge(3)),:c*6] ),false) == false

    @test TensorFunctions.issimpletensor(:( foo[:a,:b,:c] )) == true
    @test TensorFunctions.issimpletensor(:( foo(bar)[:a,:b,:c] )) == true
    @test TensorFunctions.issimpletensor(:( foo(bar)[(:a,:b),:c] )) == false
    @test TensorFunctions.issimpletensor(:( foo(bar)[(:a|hoge(2),:b),:c] )) == false
    @test TensorFunctions.issimpletensor(:( foo(bar|>hugo)[(:a|4,:b|hoge(3)),:c*6] )) == false
    @test TensorFunctions.issimpletensor(:( [:a,:b,:c] )) == false
    @test TensorFunctions.issimpletensor(:( [:a,(:b,:c)] )) == false
    @test TensorFunctions.issimpletensor(:( [(:a|4,:b|hoge(3)),:c*6] )) == false

    @test TensorFunctions.istensorproduct(:( A[:a,:b] * B[:b,:c] * C[:c,:d] )) == true
    @test TensorFunctions.istensorproduct(:( (A[:a,:b] * B[:b,:c]) * C[:c,:d] )) == false
    @test TensorFunctions.istensorproduct(:( (A[:a,:b] * B[:b,:c])[:a,:c] * C[:c,:d] )) == true
    @test TensorFunctions.istensorproduct(:( A[:a,:b] * B[:b,:c] * C[(:c,:d)] )) == true
    @test TensorFunctions.istensorproduct(:( (A[:a,:b] * B[:b,:c]) * C[(:c,:d)] )) == false
    @test TensorFunctions.istensorproduct(:( (A[:a,:b] * B[:b,:c])[:a,:c] * C[(:c,:d)] )) == true
end

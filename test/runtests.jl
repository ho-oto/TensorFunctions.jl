using TensorFunctions
using Test

@testset "TensorFunctions.jl" begin
    @test TensorFunctions.isqnode(:( :b )) == true
    @test TensorFunctions.isqnode(:( b )) == false

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

    @test TensorFunctions.isintindexproduct(:( :a*5 )) == true
    @test TensorFunctions.isintindexproduct(:( :a*hoge() )) == false
    @test TensorFunctions.isintindexproduct(:( :a )) == false
    @test TensorFunctions.isintindexproduct(:( :a*5*5 )) == false

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

    @test TensorFunctions.toindint(:( :a*5 )) == (:(:a),5)
    @test TensorFunctions.toindint(:( 5*:a )) == (:(:a),5)

    @test TensorFunctions.toname(:( A[5*:a,:c,(:d,:e)] )) == :A
    @test TensorFunctions.toindex(:( A[:a,:c,:d,:e] )) == [:(:a),:(:c),:(:d),:(:e)]
    @test TensorFunctions.toindex(:( [:a,:c,:d,:e] )) == [:(:a),:(:c),:(:d),:(:e)]

end

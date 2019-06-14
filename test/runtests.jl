using TensorFunctions
using Test
using TensorOperations

@testset "TensorFunctions.jl" begin
    @test TensorFunctions.isqnode(:( :b )) == true
    @test TensorFunctions.isqnode(:( b )) == false

    @test TensorFunctions.isqnodewithdim(:( :a )) == false
    @test TensorFunctions.isqnodewithdim(:( :b|1 )) == true
    @test TensorFunctions.isqnodewithdim(:( :b|size(foo,1) )) == true

    @test TensorFunctions.isintindexproduct(:( :a*5 )) == true
    @test TensorFunctions.isintindexproduct(:( :a*foo() )) == false
    @test TensorFunctions.isintindexproduct(:( :a )) == false
    @test TensorFunctions.isintindexproduct(:( :a*5*5 )) == false

    @test TensorFunctions.isrhspairedindex(:( (:a,:b|1,:c) )) == true
    @test TensorFunctions.isrhspairedindex(:( (:a,:b|foo(),:c) )) == true
    @test TensorFunctions.isrhspairedindex(:( (:a,:c) )) == true

    @test TensorFunctions.islhspairedindex(:( (:a,:b|1,:c) )) == false
    @test TensorFunctions.islhspairedindex(:( (:a,:b|foo(),:c) )) == false
    @test TensorFunctions.islhspairedindex(:( (:a,:c) )) == true

    @test TensorFunctions.isrhsindex(:( (:a,:b)  ))  == true
    @test TensorFunctions.isrhsindex(:( (:a,:b|bar())  ))  == true
    @test TensorFunctions.isrhsindex(:( :a*3  ))  == true
    @test TensorFunctions.isrhsindex(:( 5*:a  ))  == true
    @test TensorFunctions.isrhsindex(:( :a  ))  == true

    @test TensorFunctions.islhsindex(:( (:a,:b)  ))  == true
    @test TensorFunctions.islhsindex(:( (:a,:b|bar())  ))  == false
    @test TensorFunctions.islhsindex(:( :a*3  ))  == false
    @test TensorFunctions.islhsindex(:( 5*:a  ))  == false
    @test TensorFunctions.islhsindex(:( :a  ))  == true

    @test TensorFunctions.isrhstensor(:( foo[:a,:b,:c] )) == true
    @test TensorFunctions.isrhstensor(:( foo(bar)[:a,:b,:c] )) == true
    @test TensorFunctions.isrhstensor(:( foo(bar)[(:a,:b),:c] )) == true
    @test TensorFunctions.isrhstensor(:( foo(bar)[(:a|hoge(2),:b),:c] )) == true
    @test TensorFunctions.isrhstensor(:( foo(bar|>foo)[(:a|4,:b|hoge(3)),:c*6] )) == true
    @test TensorFunctions.isrhstensor(:( [:a,:b,:c] )) == false
    @test TensorFunctions.isrhstensor(:( [:a,(:b,:c)] )) == false
    @test TensorFunctions.isrhstensor(:( [(:a|4,:b|hoge(3)),:c*6] )) == false

    @test TensorFunctions.islhstensor(:( foo[:a,:b,:c] )) == false
    @test TensorFunctions.islhstensor(:( foo(bar)[:a,:b,:c] )) == false
    @test TensorFunctions.islhstensor(:( foo(bar)[(:a,:b),:c] )) == false
    @test TensorFunctions.islhstensor(:( foo(bar)[(:a|hoge(2),:b),:c] )) == false
    @test TensorFunctions.islhstensor(:( foo(bar|>hugo)[(:a|4,:b|hoge(3)),:c*6] )) == false
    @test TensorFunctions.islhstensor(:( [:a,:b,:c] )) == true
    @test TensorFunctions.islhstensor(:( [:a,(:b,:c)] )) == true
    @test TensorFunctions.islhstensor(:( [(:a|4,:b|hoge(3)),:c*6] )) == false

    @test TensorFunctions.isrhssimpletensor(:( foo[:a,:b,:c] )) == true
    @test TensorFunctions.isrhssimpletensor(:( foo(bar)[:a,:b,:c] )) == true
    @test TensorFunctions.isrhssimpletensor(:( foo(bar)[(:a,:b),:c] )) == false
    @test TensorFunctions.isrhssimpletensor(:( foo(bar)[(:a|hoge(2),:b),:c] )) == false
    @test TensorFunctions.isrhssimpletensor(:( foo(bar|>foo)[(:a|4,:b|hoge(3)),:c*6] )) == false
    @test TensorFunctions.isrhssimpletensor(:( [:a,:b,:c] )) == false
    @test TensorFunctions.isrhssimpletensor(:( [:a,(:b,:c)] )) == false
    @test TensorFunctions.isrhssimpletensor(:( [(:a|4,:b|hoge(3)),:c*6] )) == false

    @test TensorFunctions.islhssimpletensor(:( foo[:a,:b,:c] )) == false
    @test TensorFunctions.islhssimpletensor(:( foo(bar)[:a,:b,:c] )) == false
    @test TensorFunctions.islhssimpletensor(:( foo(bar)[(:a,:b),:c] )) == false
    @test TensorFunctions.islhssimpletensor(:( foo(bar)[(:a|hoge(2),:b),:c] )) == false
    @test TensorFunctions.islhssimpletensor(:( foo(bar|>hugo)[(:a|4,:b|hoge(3)),:c*6] )) == false
    @test TensorFunctions.islhssimpletensor(:( [:a,:b,:c] )) == true
    @test TensorFunctions.islhssimpletensor(:( [:a,(:b,:c)] )) == false
    @test TensorFunctions.islhssimpletensor(:( [(:a|4,:b|hoge(3)),:c*6] )) == false

    @test TensorFunctions.istensorproduct(:( A[:a,:b] * B[:a,:c,:b] * C[:d,:r] )) == true
    @test TensorFunctions.istensorproduct(:( A[:a,:b] * B[:a,(:c,:b)] * C[:d*1000,:r] )) == true
    @test TensorFunctions.istensorproduct(:( 1 * A[:a,:b] * B[:a,(:c,:b)] * C[:d*1000,:r] )) == false
    @test TensorFunctions.istensorproduct(:( foo() * A[:a,:b] * B[:a,(:c,:b)] * C[:d*1000,:r] )) == false
    @test TensorFunctions.istensorproduct(:( (A[:a,:b] * B[:a,(:c,:b)]) * C[:d*1000,:r] )) == false
    @test TensorFunctions.istensorproduct(:( (A[:a,:b] * B[:a,(:c,:b)])[:A,:B] * C[:d*1000,:r] )) == true

    @test TensorFunctions.issimpletensorproduct(:( A[:a,:b] * B[:a,:c,:b] * C[:d,:r] )) == true
    @test TensorFunctions.issimpletensorproduct(:( A[:a,:b] * B[:a,(:c,:b)] * C[:d*1000,:r] )) == false
    @test TensorFunctions.issimpletensorproduct(:( 1 * A[:a,:b] * B[:a,(:c,:b)] * C[:d*1000,:r] )) == false
    @test TensorFunctions.issimpletensorproduct(:( foo() * A[:a,:b] * B[:a,(:c,:b)] * C[:d*1000,:r] )) == false
    @test TensorFunctions.issimpletensorproduct(:( (A[:a,:b] * B[:a,(:c,:b)]) * C[:d*1000,:r] )) == false
    @test TensorFunctions.issimpletensorproduct(:( (A[:a,:b] * B[:a,(:c,:b)])[:A,:B] * C[:d,:r] )) == true

    A = randn(3,4,5,6,7,8,3,3); B = randn(4,5,2)
    @tensor C[a,c,e,f,h] := reshape(A,3, 2,2 ,5 ,6*7, 2,4 ,3*3)[a,b,c,d,e,f,g,h] * B[g,d,b]
    C = reshape(C,:,size(C)[3:end]...)
    C2 = @tensorfunc [(:a,:c),:e,:f,:h] <= A[:a,(:b,:c),:d,:e*2,(:f,:g),:h*2] * B[:g,:d,:b]
    @test isapprox(C, C2)

    A = randn(100); B= randn(5,5)
    @tensor C[a,bb,c] := reshape(A,2,5,10)[a,b,c] * B[b,bb]
    C2 = @tensorfunc [:a,:bb,:c] <= A[(:a,:b,:c|10)] * B[:b,:bb]
    @test isapprox(C, C2)

end

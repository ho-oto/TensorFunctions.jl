using TensorFunctions
using Test
using TensorOperations
using TensorCast

@testset "TensorFunctions.jl" begin
    function f(A,B)
        @cast Aprime[a,b,c,e] := A[a,(b,c),e] b:5
        @cast Bprime[c,d,e] := B[c,d,e]
        @tensor tmp[-1,-2,-3] := Aprime[-1,-2,2,1] * Bprime[2,-3,1]
        @cast res[(a,b),d] := tmp[a,b,d]
    end
    g(A,B) = @tensorfunc [(:a,:b),:d,:e] (A[:a,(:b,:c),:e],(:b,2)) * B[:c,:d,:e] (:c=2,:e=1)
    A = randn(4,25,6); B = randn(5,7,6)
    @assert f(A,B) == g(A,B)
end

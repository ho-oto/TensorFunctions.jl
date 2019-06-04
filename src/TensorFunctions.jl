module TensorFunctions

using TensorOperations
using TensorCast

export @tensorfunc

"""
f(A,B) = @tensorfunc[(:a,:b),:d,:e] (:c=2,:e=1) (A[:a,(:b,:c),:e] :b=2) * B[:c,:d,:e]
<=>
function f(A,B)
    @cast Aprime[a,b,c,e] := A[a,(b,c),e] b:2
    @cast Bprime[c,d,e] := B[c,d,e]
    @tensor tmp[-1,-2,-3] := Aprime[-1,-2,2,1] * Bprime[2,-3,1]
    @cast res[(a,b),d] := tmp[a,b,d]
end
"""
macro tensorfunc(ex::Expr)
    ex
end

end # module

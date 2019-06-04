module TensorFunctions

using TensorOperations
using LinearMaps
using TensorCast

export @tensorfunc

'''
f(A,B) = @tensorfunc[(:a,:b),:d] (:a=1,:b=3,:c=2) (A[:a,(:b,:c)] :b=2) * B[:c,:d]
<=>
function f(A,B)
    @cast Aprime[a,b,c] := A[a,(b,c)] b:2
    @cast Bprime[c,d] := B[c,d]
    @tensoropt (a=>chi,b=>chi^2,c=>chi) tmp[a,b,d] := Aprime[a,b,c] * Bprime[c,d]
    @cast res[(a,b),d] = tmp[a,b,d]
end


'''
macro tensorfunc(ex::Expr)
    ex
end

end # module

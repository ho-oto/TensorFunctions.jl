module TensorFunctions

using TensorOperations
using TensorCast

export @tensorfunc

"""
f(A,B) = @tensorfunc[(:a,:b),:d,:e] (A[:a,(:b,:c),:e] :b 2) * B[:c,:d,:e] (:c=2,:e=1)
<=>
function f(A,B)
    @cast Aprime[a,b,c,e] := A[a,(b,c),e] b:2
    @cast Bprime[c,d,e] := B[c,d,e]
    @tensor tmp[-1,-2,-3] := Aprime[-1,-2,2,1] * Bprime[2,-3,1]
    @cast res[(a,b),d] := tmp[a,b,d]
end
"""
macro tensorfunc(ind::Expr,ex::Expr,ncon::Expr)
    ntensors = ex.args |> length
    outex = Expr(:block)
    for i in 1:ntensors
        tmpex = 0
        push!(outex.args,tmpex)
    end
end

function quotenode_to_symbol(ex::Expr)
    if !(ex.head == :ref); error("ex should be :(Foo[:a,(:b,:c)])"); end
    exout = Expr(:ref,ex.args[1])
    for i in ex.args[2:end]
        if typeof(i) == QuoteNode
            push!(exout.args,i|>eval)
        elseif typeof(i) == Expr
            if !(i.head == :tuple); error("combined index should be expressed by ()"); end
            extmp = Expr(:tuple)
            for j in i.args
                push!(extmp.args,j|>eval)
            end
            push!(exout.args,extmp)
        end
    end
    exout
end

end # module

module TensorFunctions

using LinearAlgebra,TensorOperations

export @tensorfunc

#= global setting =#
function order(ex::Expr,ord::Expr)
    if ord.head == :tuple
        QuoteNode[ord.args...]
    else
        error("not implemented")
    end
end
function order(ex::Expr)
    (nothing,)
end

tracefunc=tensortrace
contractfunc=tensorcontract

include("tensorfunc.jl")

end # module

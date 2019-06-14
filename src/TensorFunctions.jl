module TensorFunctions

using LinearAlgebra,TensorOperations

export @tensorfunc

#= global setting =#
function order(ex::Expr)
    if ex.head == :tuple
        QuoteNode.(ex.args)
    else
        error("not implemented")
    end
end
function order()
    (nothing,)
end
tracefunc=tensortrace
contractfunc=tensorcontract
#= end global setting =#

include("tensorfunc.jl")

end # module

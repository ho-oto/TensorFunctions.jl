module TensorFunctions

using LinearAlgebra,TensorOperations

export @tensorfunc

#= global setting =#
function order(ex::Expr)
    if ex.head == :tuple
        ex.args
    else
        error("not implemented")
    end
end
tracefunc=tensortrace
contractor=tensorcontract
#= end global setting =#

include("tensorfunc.jl")

end # module

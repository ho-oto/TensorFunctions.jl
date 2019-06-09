module TensorFunctions

using LinearAlgebra
using TensorOperations: contract!,trace!

export @tensorfunc,@tensormap

function issymbol(ex)
    if typeof(ex) != QuoteNode
        false
    else
        true
    end
end

issymbol(ex) = typeof(ex) == QuoteNode ? true : false
tosymbol(ex) = issymbol(ex) ? ex.value : nothing

function istensor(ex)
    if typeof(ex) != Expr
        false
    elseif ex.head != :ref
        false
    elseif all(typeof.(ex.args[2:end]) .== QuoteNode)
        true
    end
end
function totensor(ex)
    if istensor(ex)
        ex.args[1],tosymbol.(ex.args[2:end])
    else
        nothing,nothing
    end
end

end # module

module TensorFunctions

using LinearAlgebra
using TensorOperations: contract!,trace!

export @tensorfunc,@tensormap

issymbol(ex) = typeof(ex) == QuoteNode ? true : false
tosymbol(ex) = issymbol(ex) ? ex.value : nothing

function isinpairedindex(ex,lorr)
    if lorr == :rhs
        if typeof(ex) == Expr
            ex.head == :call &&
                ex.args[1] == :| &&
                issymbol(ex.args[2]) &&
                length(ex.args) == 3
        else
            issymbol(ex)
        end
    elseif lorr == :lhs
        issymbol(ex)
    else
        error(":lhs or :rhs")
    end
end

function ispairedindex(ex,lorr)
    if typeof(ex) != Expr
        false
    elseif ex.head != :tuple
        false
    else
        all(ex.args .|> x -> isinpairedindex(x,lorr))
    end
end

function isindexproduct(ex)
    if typeof(ex) != Expr
        false
    elseif ex.head == :call &&
        ex.args[1] == :* &&
        issymbol(ex.args[2]) &&
        typeof(ex.args[3]) == Int &&
        length(ex.args) == 3
        true
    else
        false
    end
end

isindex(ex,lorr) = issymbol(ex) || ispairedindex(ex,lorr)

function istensor(ex,lorr)
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

function istensorproduct(ex)
    if typeof(ex) != Expr
        false
    elseif ex.head == :ref && ex.args[1] == :*
        true
    else
        false
    end
end

function lhsandrhs(ex::Expr)
    if ex.head in [:(=),:(:=),:(<=),:(+=),:(-=)]
        if istensor(ex.args[1],:lhs)
            return ex.args[1],ex.args[2]
        else
            error("parse error")
        end
    elseif ex.head == :(=>)
        if istensor(ex.args[2],:lhs)
            return ex.args[2],ex.args[1]
        else
            error("parse error")
        end
    else
        error("parse error")
    end
end

end # module

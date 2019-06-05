module TensorFunctions

using TensorOperations
using TensorCast

function remove_quote(ex::Expr)
    if ex.head == :ref
        exout = Expr(:ref,ex.args[1])
        for elem in ex.args[2:end]
            if typeof(elem) == QuoteNode
                push!(exout.args,elem|>eval)
            elseif typeof(elem) == Expr && elem.head == :tuple
                push!(exout.args,remove_quote(elem))
            else
                error("ex.args should be Symbol or tuple of Symbols")
            end
        end
    elseif ex.head == :vect || ex.head == :tuple
        exout = Expr(ex.head)
        for elem in ex.args
            if typeof(elem) == QuoteNode
                push!(exout.args,elem|>eval)
            elseif typeof(elem) == Expr && elem.head == :tuple
                push!(exout.args,remove_quote(elem))
            else
                error("ex.args should be Symbol or tuple of Symbols")
            end
        end
    else
        error("ex.head should be [:ref,:vect,:tuple]")
    end
    exout
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

function quotenode_to_symbol_(ex::Expr,name::Symbol)
    if !(ex.head == :vect); error("ex should be :([:a,(:b,:c)])"); end
    exout = Expr(:ref,name)
    for i in ex.args
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

function remove_bracket(ex::Expr,name::Symbol)
    exout = Expr(:ref)
    push!(exout.args,name)
    for i in ex.args[2:end]
        if typeof(i) == Symbol
            push!(exout.args,i)
        elseif typeof(i) == Expr && i.head == :tuple
            for j in i.args
                push!(exout.args,j)
            end
        else
            error("?????")
        end
    end
    exout
end

remove_bracket(ex::Expr) = remove_bracket(ex,ex.args[1])

function remove_bracket_(ex::Expr,name::Symbol)
    exout = Expr(:ref,name)
    for i in ex.args
        if typeof(i|>eval) == Symbol
            push!(exout.args,i|>eval)
        elseif typeof(i) == Expr && i.head == :tuple
            for j in i.args
                push!(exout.args,j|>eval)
            end
        else
            error("?????")
        end
    end
    exout
end

set_size(ex::Expr) = Expr(:call,Symbol(":"),ex.args[1]|>eval,ex.args[2])

function _tensorfunc(ind::Expr,ex::Expr,ncon::Expr)
    if !(ind.head == :vect); error("expected to be :vect"); end
    if !(ex.head == :call); error("expected to be :call"); end
    if !(ex.args[1] == :*); error("only * is supported"); end
    if !(ncon.head == :tuple); error("expected to be :tuple"); end

    outex = Expr(:block)
    exx = copy(ex)

    ncondict = Dict((ncon.args[i]|>eval,i) for i in 1:length(ncon.args))
    inddict = Dict((ind.args[i]|>eval,-i) for i in 1:length(ind.args))
    merge!(ncondict,inddict)

    for i in 2:length(ex.args)
        if ex.args[i].head == :ref
            exx.args[i] = quotenode_to_symbol(ex.args[i])
            for k in 2:length(exx.args[i].args)
                exx.args[i].args[k] = ncondict[exx.args[i].args[k]]
            end
        elseif ex.args[i].head == :tuple
            newsym = gensym()
            rhs = quotenode_to_symbol(ex.args[i].args[1])
            lhs = remove_bracket(rhs,newsym)
            tmpargs = [set_size(j) for j in ex.args[i].args[2:end]]
            push!(outex.args,TensorCast._macro(Expr(:(:=),lhs,rhs),tmpargs...))
            exx.args[i] = lhs
            for k in 2:length(exx.args[i].args)
                exx.args[i].args[k] = ncondict[exx.args[i].args[k]]
            end
        else
            error("???")
        end
    end
    newsym = gensym()
    push!(outex.args,TensorOperations.tensorify(Expr(:(:=),Expr(:ref,newsym,Symbol(":")),exx)))
    ressym = gensym()
    push!(outex.args,TensorCast._macro(Expr(:(:=),quotenode_to_symbol_(ind,ressym),remove_bracket_(ind,newsym))))
    outex
end

function _tensorfunc(ind::Expr,ex::Expr)
    if !(ind.head == :vect); error("expected to be :vect"); end
    if !(ex.head == :call); error("expected to be :call"); end
    if !(ex.args[1] == :*); error("only * is supported"); end

    outex = Expr(:block)
    exx = copy(ex)

    for i in 2:length(ex.args)
        if ex.args[i].head == :ref
            exx.args[i] = quotenode_to_symbol(ex.args[i])
        elseif ex.args[i].head == :tuple
            newsym = gensym()
            rhs = quotenode_to_symbol(ex.args[i].args[1])
            lhs = remove_bracket(rhs,newsym)
            tmpargs = [set_size(j) for j in ex.args[i].args[2:end]]
            push!(outex.args,TensorCast._macro(Expr(:(:=),lhs,rhs),tmpargs...))
            exx.args[i] = lhs
        else
            error("???")
        end
    end
    newsym = gensym()
    push!(outex.args,TensorOperations.tensorify(Expr(:(:=),remove_bracket_(ind,newsym),exx)))
    ressym = gensym()
    push!(outex.args,TensorCast._macro(Expr(:(:=),quotenode_to_symbol_(ind,ressym),remove_bracket_(ind,newsym))))
    outex
end

#export @tensorfunc

end # module

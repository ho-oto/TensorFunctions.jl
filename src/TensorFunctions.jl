module TensorFunctions

using TensorOperations
using TensorCast

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

"""
f(A,B) = @tensorfunc[(:a,:b),:d,:e] (A[:a,(:b,:c),:e],(:b,2)) * B[:c,:d,:e] (:e,:c)
<=>
function f(A,B)
    @cast Aprime[a,b,c,e] := A[a,(b,c),e] b:2
    @cast Bprime[c,d,e] := B[c,d,e]
    @tensor tmp[-1,-2,-3] := Aprime[-1,-2,2,1] * Bprime[2,-3,1]
    @cast res[(a,b),d] := tmp[a,b,d]
end
"""
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

#export @tensorfunc

end # module

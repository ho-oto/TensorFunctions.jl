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
            if typeof(elem) == QuoteNode && typeof(elem|>eval) == Symbol
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

function remove_bracket(ex::Expr)
    havebracket = false
    if ex.head == :ref
        exout = Expr(:ref,ex.args[1])
        for elem in ex.args[2:end]
            if (typeof(elem) == QuoteNode && typeof(elem|>eval) == Symbol) || typeof(elem) == Symbol
                push!(exout.args,elem)
            elseif typeof(elem) == Expr && elem.head == :tuple
                push!(exout.args,elem.args...)
                havebracket = true
            else
                error("ex.args should be Symbol or QuotedSymbol tuple of [Symbols,QuotedSymbols]")
            end
        end
    elseif ex.head == :vect || ex.head == :tuple
        exout = Expr(ex.head)
        for elem in ex.args
            if (typeof(elem) == QuoteNode && typeof(elem|>eval) == Symbol) || typeof(elem) == Symbol
                push!(exout.args,elem)
            elseif typeof(elem) == Expr && elem.head == :tuple
                push!(exout.args,elem.args...)
                havebracket = true
            else
                error("ex.args should be Symbol or QuotedSymbol tuple of [Symbols,QuotedSymbols]")
            end
        end
    else
        error("ex.head should be [:ref,:vect,:tuple]")
    end
    exout,havebracket
end

function give_name(ex::Expr,name::Symbol)
    if ex.head == :ref
        exout = copy(ex)
        exout.args[1] = name
    elseif ex.head == :vect || ex.head == :tuple
        exout = Expr(:ref,name)
        push!(exout.args,ex.args...)
    else
        error("ex.head should be [:ref,:vect,:tuple]")
    end
    exout
end

function set_size(ex::Expr)
    if ex.head == :tuple && (ex.args[1]|>eval|>typeof) == Symbol && typeof(ex.args[2]) == Int && length(ex.args) == 2
        Expr(:call,Symbol(":"),ex.args[1]|>eval,ex.args[2])
    else
        error("ex should be :((:hoge,huga::Int))")
    end
end

function symbol_to_int(ex::Expr,inddict::Dict{Symbol,Int})
    #TODO: enable to parse nested one like A[(:a,:b),:c]
    exout = copy(ex)
    if ex.head == :ref
        for i in 2:length(ex.args)
            exout.args[i] = inddict[ex.args[i]|>eval]
        end
    elseif ex.head == :vect || ex.head == :tuple
        for i in 1:length(ex.args)
            exout.args[i] = inddict[ex.args[i]|>eval]
        end
    else
        error("ex.head should be [:ref,:vect,:tuple]")
    end
    exout
    
end

function _tensorfunc(ex::Expr)
    #TODO: enable to parse nested one like (A[:a,:b] * B[:b,:c]) * C[:c,:d]
    if ex.head == :call && ex.args[1] == :(=>) && length(ex.args) == 3
        inp,outp = ex.args[2:3]
    elseif ex.head == :call && ex.args[1] == :(<=) && length(ex.args) == 3
        outp,inp = ex.args[2:3]
    else
        error("parse fails")
    end
    if outp.head != :vect || inp.head != :call || inp.args[1] != :*
        error("parse fails")
    end
    exout = Expr(:block)
    for elem in inp.args[2:end]
        if elem.head == :tuple && elem.args[1].head == :ref
    end
    ###
end

"""
    f(A,B,C) = @tensorfunc [(:a,:b),(:c,:d)] <= A[:a,:x] * (B[(:x,:y)],(:x,5)) * (C[(:y,:b,:c),:d],(:y,6),(:b,7)) options
    f(A,B,C) = @tensorfunc A[:a,:x] * (B[(:x,:y)],(:x,5)) * (C[(:y,:b,:c),:d],(:y,6),(:b,7)) => [(:a,:b),(:c,:d)] options
"""
macro tensorfunc(ex::Expr)
    _tensorfunc(ex)
end

#export @tensorfunc

end # module

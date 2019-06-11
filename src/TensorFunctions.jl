module TensorFunctions

using LinearAlgebra,TensorOperations

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

isindex(ex,lorr) = issymbol(ex) || ispairedindex(ex,lorr) || isindexproduct(ex)

function istensor(ex,lorr)
    if typeof(ex) != Expr
        false
    elseif ex.head == :ref
        all(ex.args[2:end] .|> x -> isindex(x,lorr))
    elseif ex.head == :vect
        if lorr != :lhs
            false
        else
            all(ex.args .|> x -> isindex(x,lorr))
        end
    else
        false
    end
end

function tosimpletensor(ex,arg::Dict{Symbol,Int})
    if !istensor(ex,:rhs)
        error("not tensor")
    else
        tensorname = ex.args[1]
        indexlist = ex.args[2:end]
        newindexlist = QuoteNode[]
        for i in indexlist
            if issymbol(i)
                push!(newindexlist,i)
            else

            end
        end
    end
end

function totensor(ex)
    if istensor(ex)
        if ex.head == :vect
            nothing,ex.args
        else
            ex.args[1],ex.args[2:end]
        end
    else
        nothing,nothing
    end
end

function istensorproduct(ex)
    if typeof(ex) != Expr
        false
    elseif ex.head == :call &&
        ex.args[1] == :* &&
        all(ex.args[2:end] .|> x -> istensor(x,:rhs))
        true
    else
        false
    end
end

function lhsrhs(ex::Expr)
    if length(ex.args) != 2
        error("parse error")
    end
    if ex.head in [:(=),:(:=),:(<=),:(+=),:(-=)]
        if istensor(ex.args[1],:lhs) && istensorproduct(ex.args[2])
            return ex.head,ex.args[1],ex.args[2]
        else
            error("parse error")
        end
    elseif ex.head == :(=>)
        if istensor(ex.args[2],:lhs) && istensorproduct(ex.args[1])
            return ex.head,ex.args[2],ex.args[1]
        else
            error("parse error")
        end
    else
        error("parse error")
    end
end

function parsetensorproduct(ex)
    # (foo[a,b] * bar[b,c])[a,c] ->
    # quote
    # Foo = parsetensorproduct(foo[a,b])
    # Bar = parsetensorproduct(bar[b,c])
    # tensorcontract(Foo,Index1,Bar,Index2,Index3)
    # end
    if !istensor(ex)
        error("ex should be tensor")
    else

    end
end

function tensorproductmain(ex,contractorder)
    # 1. A[(:a,:b),:c*5] -> reshape(A,...)[:a,:b,:c]
    # 2. convert to NCON
    # 3. A*B*C*D*E -> (( A[foo,bar] * B[bar,hoge] )[foo,hoge] * ( C[...] * ( D[...] * E[...] ))[...])[hoge,huga]
    #=
    (( A * B ) * ( C * ( D * E )))
    ->
    tmp1 = (D*E)
    (( A * B ) * ( C * tmp1))
    ->
    tmp1 = (D*E)
    tmp2 = (C*tmp1)
    tmp3 = (A*B)
    (tmp3 * tmp2)
    =#
    # 4. reshape the result

    #= step 1 =#
    head,lhs,rhs = lhsrhs(ex)

    return quote
        reslhs = resrhs
    end
end

function tensormapmain(ex::Expr)

end

macro tensorfunc(ex::Expr)
    tensorproductmain(ex)
end

macro tensormap(ex::Expr)
    tensormapmain(ex)
end

end # module

module TensorFunctions

using LinearAlgebra,TensorOperations

export @tensorfunc#,@tensormap

issymbol(ex) = typeof(ex) == QuoteNode ? true : false

function isinpairedindex(ex,lorr) # :a|hoge,:a -> true
    if lorr == :rhs
        if typeof(ex) == Expr
            ex.head == :call && ex.args[1] == :| && issymbol(ex.args[2]) && length(ex.args) == 3
        else
            issymbol(ex)
        end
    elseif lorr == :lhs
        issymbol(ex)
    else
        error(":lhs or :rhs")
    end
end

ispairedindex(ex,lorr) = false
ispairedindex(ex::Expr,lorr) = (ex.head == :tuple) && all(ex.args .|> x -> isinpairedindex(x,lorr))

isindexproduct(ex) = false
isindexproduct(ex::Expr) = ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 &&
    ex.args[2:3].|>typeof|>Set == (Int,QuoteNode)|>Set

isindex(ex,lorr) = issymbol(ex) || ispairedindex(ex,lorr) || (lorr == :rhs && isindexproduct(ex))

istensor(ex,lorr) = false
istensor(ex::Expr,lorr) = (ex.head == :ref && all(ex.args[2:end] .|> x -> isindex(x,lorr))) ||
    (ex.head == :vect && all(ex.args .|> x -> isindex(x,lorr)) && (lorr == :lhs))

function tosimpletensor(ex,arg::Dict{Symbol,Int})
    # (hoge)[:a,:b,(:c,:d),:e*5] -> reshape(trace(hoge))[:a,:b,:c,:d,:e]
    # TODO:
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

function tonameindex(ex,lorr) # A[:a,:b] -> :A,[:(:a),:(:b)]
    if istensor(ex,lorr)
        if ex.head == :vect
            nothing,ex.args
        else
            ex.args[1],ex.args[2:end]
        end
    else
        error("not tensor")
    end
end

function istensorproduct(ex) # A[:a,:b] * (B[:b,:c] * C[(:c,:d)]) -> true
    if typeof(ex) != Expr
        false
    elseif ex.head == :call &&
        ex.args[1] == :* &&
        all(ex.args[2:end] .|> x -> istensorproduct(x,:rhs))
        true
    elseif istensor(ex,:rhs)
        true
    else
        false
    end
end

function issimpletensorproduct(ex)
    if typeof(ex) != Expr
        false
    else
        ex.head == :call && ex.args[1] == :* && all(ex.args[2:end] .|> x -> istensor(x,:rhs))
    end
end

function toheadlhsrhs(ex::Expr) # hoge = huga -> :=,hoge,huga
    if length(ex.args) == 2
        if ex.head in [:(=),:(:=),:(+=),:(-=)]
            if istensor(ex.args[1],:lhs) && istensorproduct(ex.args[2])
                return ex.head,ex.args[1],ex.args[2]
            else
                error("parse error")
            end
        else
            error("parse error")
        end
    elseif length(ex.args) == 3
        if ex.head == :call && ex.args[1] == :(<=)
            if istensor(ex.args[2],:lhs) && istensorproduct(ex.args[3])
                return ex.args[1],ex.args[2],ex.args[3]
            else
                error("parse error")
            end
        elseif ex.head == :call && ex.args[1] == :(=>)
            if istensor(ex.args[3],:lhs) && istensorproduct(ex.args[2])
                return ex.args[1],ex.args[3],ex.args[2]
            else
                error("parse error")
            end
        end
    else
        error("parse error")
    end
end

function parsetensorproduct(ex,contractor=tensorcontract)
    #TODO: use TensorOperator.contract_indices at compile time
    if !istensor(ex,:rhs)
        error("ex should be tensor")
    elseif istensorproduct(ex.args[1])
        lhs = gensym()
        rhs = gensym()
        lhs2 = parsetensorproduct(ex.args[1].args[2])
        rhs2 = parsetensorproduct(ex.args[1].args[3])
        return quote
            $lhs = $lhs2
            $rhs = $rhs2
            $contractor($lhs,
                $(ex.args[1].args[2].args[2:end]|>Tuple),
                $rhs,
                $(ex.args[1].args[3].args[2:end]|>Tuple),
                $(ex.args[2:end]|>Tuple))
        end
    else
        :($(ex.args[1]))
    end
end

function nonduplicateindex(indslis)
    res_ = [indslis[1]...]
    res = eltype(res_)[]
    for i in res_
        if !(i in res)
            push!(res,i)
        end
    end
    for inds in indslis[2:end]
        for ind in inds
            if ind in res
                filter!(x->x!=ind,res)
            else
                push!(res,ind)
            end
        end
    end
    res
end

haveduplicatedindex(inds) = (inds|>Set|>length) != (inds|>length)

function makepairwised(ex::Expr,contractorder)
    if ex.head != :call || ex.args[1] != :*
        error("parse error")
    elseif length(ex.args) == 3 #2コの積
        ex
    else
        # 露出しているindexの中で一若いのをペアにして全体を自分に食わせる
        exx = copy(ex)
        indslis = [i.args[2:end] for i in ex.args[2:end]]
        tmp = Int[]
        for j in contractorder
            for k in 1:length(indslis)
                if j in indslis[k]; push!(tmp,k); end
            end
            if length(tmp) == 2; break; end
        end
        filter!(x->!(x in tmp),exx.args[2:end])
        push!(exx.args,Expr(:ref,Expr(:call,:*,ex.args[],ex.args[]),nonduplicateindex()...))
        makepairwised(exx)
    end
end

function tensorproductmain(ex,contractorder=nothing)
    # 0. calculate dimension list (dim(:a) = ...,)
    # 1. A[(:a,:b),:c*5,:d,:d] -> reshape(trace(A),...)[:a,:b,:c] (tosimpletensor)
    # 3. A*B*C*D*E -> (( A[foo,bar] * B[bar,hoge] )[foo,hoge] * ( C[...] * ( D[...] * E[...] ))[...])[hoge,huga]
    # 4. reshape the result
    head,lhs,rhs = toheadlhsrhs(ex)
    rhs = tosimpletensor(rhs)
    rhs = taketensor(rhs)
    rhs = makepairwised(rhs,contractorder)
    rhs = Expr(:ref,rhs)
    append!(rhs.args,tonameindex(lhs,:lhs)[2])
    rhs = parsetensorproduct(rhs)
    if head == :(<=) || head == :(=>)
        rhs
    else
        lhsname = tonameindex(lhs,:lhs)[1]
        op = if head == :(:=)
            :(=)
        elseif head == :(=)
            :(.=)
        elseif head == :(+=)
            :(.+=)
        elseif head == :(-=)
            :(.-=)
        end
        Expr(op,lhsname,rhs)
    end
end

macro tensorfunc(ex::Expr)
    esc(tensorproductmain(ex))
end


#= TODO: implement
function tensormapmain(ex::Expr)
    ex
end

macro tensormap(ex::Expr)
    esc(tensormapmain(ex))
end
=#

end # module

module TensorFunctions

using LinearAlgebra,TensorOperations

export @tensorfunc#,@tensormap

#= Bool functions =#
issymbol(ex) = typeof(ex) == QuoteNode

isinpairedindex(ex,inrhs::Bool=true) = false
isinpairedindex(ex::QuoteNode,inrhs::Bool=true) = true
isinpairedindex(ex::Expr,inrhs::Bool=true) = inrhs && ex.head == :call &&
    ex.args[1] == :| && issymbol(ex.args[2]) && length(ex.args) == 3

ispairedindex(ex,inrhs::Bool=true) = false
ispairedindex(ex::Expr,inrhs::Bool=true) = (ex.head == :tuple) &&
    all(ex.args .|> x -> isinpairedindex(x,inrhs))

isindexproduct(ex) = false
isindexproduct(ex::Expr) = ex.head == :call && ex.args[1] == :* &&
    length(ex.args) == 3 && ex.args[2:3].|>typeof|>Set == (Int,QuoteNode)|>Set

isindex(ex,inrhs::Bool=true) = issymbol(ex) || ispairedindex(ex,inrhs) ||
    (inrhs && isindexproduct(ex))

istensor(ex,inrhs::Bool=true) = false
istensor(ex::Expr,inrhs::Bool=true) =
    (ex.head == :ref && all(ex.args[2:end] .|> x -> isindex(x,inrhs))) ||
    (ex.head == :vect && all(ex.args .|> x -> isindex(x,inrhs)) && !inrhs)

issimpletensor(ex) = false
issimpletensor(ex::Expr) = ex.head == :ref && all(ex.args[2:end] .|> issymbol)

istensorproduct(ex) = false
istensorproduct(ex::Expr) = ex.head == :call && ex.args[1] == :* &&
    all(ex.args[2:end] .|> x -> istensor(x))

issimpletensorproduct(ex) = false
issimpletensorproduct(ex::Expr) = ex.head == :call && ex.args[1] == :* &&
    all(ex.args[2:end] .|> x -> issimpletensor(x))
#= end Bool functions =#

#= elementary functions for parser =#
function tonameindex(ex::Expr) # A[:a,:b] -> :A,[:(:a),:(:b)]
    if     ex.head == :vect ; nothing,ex.args
    elseif ex.head == :ref  ; ex.args[1],ex.args[2:end]
    else                    ; error("not tensor")
    end
end

function toheadlhsrhs(ex::Expr)
    if ex.head in [:(=),:(:=),:(+=),:(-=)] && istensor(ex.args[1],false) &&
        istensorproduct(ex.args[2]) && length(ex.args) == 2
        ex.head,ex.args[1],ex.args[2]
    elseif ex.head == :call && ex.args[1] == :(<=) && istensor(ex.args[2],false) &&
        istensorproduct(ex.args[3]) && length(ex.args) == 3
        ex.args[1],ex.args[2],ex.args[3]
    elseif ex.head == :call && ex.args[1] == :(=>) && istensor(ex.args[3],false) &&
        istensorproduct(ex.args[2]) && length(ex.args) == 3
        ex.args[1],ex.args[3],ex.args[2]
    else
        error("parse error")
    end
end

function duplicateindex(indslis::Array{<:Array{<:Any,1},1})
    res = Any[]
    dup = Any[]
    for inds in indslis
        for ind in inds
            if ind in dup
                error("same index appears more than two times")
            elseif ind in res
                filter!(x->x!=ind,res)
                push!(dup,ind)
            else
                push!(res,ind)
            end
        end
    end
    res,dup
end
function duplicateindex(ex::Expr)
    if issimpletensorproduct(ex)
        indslis = [i.args[2:end] for i in ex.args[2:end]]
        duplicateindex(indslis)
    else
        error("ex should be product of tensors")
    end
end
#= elementary functions for parse =#

function tosimpletensor(ex,arg::Dict{Symbol,Int})
    if istensorproduct(ex)
        exx = copy(ex)
        for i in 2:length(ex.args)
            exx.args[i] = tosimpletensor(ex.args[i])
        end
        exx
    elseif istensor(ex)
        # (hoge)[:a,:b,(:c,:d),:e*5] -> reshape(trace(hoge))[:a,:b,:c,:d,:e]
        # TODO:
        if !istensor(ex)
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
end

function taketrace(ex::Expr,tracefunc=tensortrace)
    if istensorproduct(ex)
        exx = copy(ex)
        for i in 2:length(ex.args)
            exx.args[i] = taketrace(ex.args[i])
        end
        exx
    elseif istensor(ex)
        tname,tind = tonameindex(ex)
        if haveduplicatedindex([tind])
            newtind = nonduplicateindex([tind])
            :($tracefunc($tname,$tind,$newtind)$newtind)
        else
            ex
        end

    end
end

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


function parsetensorproduct(ex,contractor=tensorcontract)
    #TODO: use TensorOperator.contract_indices at compile time
    if !istensor(ex)
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



function tensorproductmain(ex,contractorder::NTuple{N,Symbol} where N)
    # reshapeの必要なtensorをリストアップする : 0コなら何もしない
    # 露出しているindexをリスト化する : :a=>size(A,1),:b=>size(B,2),:c=>prod(size(C)[[2,3,4]]),...
    # 明示的に決まっているやつをリストに足す
    # ()の中で不明なのが残っていれば残りから決める : :d = div(size(D,1),(hoge*huga*piyo))
    # 元のExprから何も考えずに()と*Intを外したものを作る
    # 愚直にreshapeする
    # TODO: reshape the result
    head,lhs,rhs = toheadlhsrhs(ex) # rhs = A[:a,:b] * B[(:b,:c|hoge)] * C[(:c,:d),:e,:e]
    rhs = tosimpletensor(rhs) # A[:a,:b] * reshape(B)[:b,:c] * reshape(C)[:c,:d,:e,:e]
    rhs = taketrace(rhs) # A[:a,:b] * reshape(B)[:b,:c] * trace(reshape(C))[:c,:d]
    rhs = makepairwised(rhs,contractorder) # (A[:a,:b] * B[:b,:c])[:a,:c] * C[:c,:d]
    rhs = Expr(:ref,rhs,tonameindex(lhs)[2]...) # ((A[:a,:b] * B[:b,:c])[:a,:c] * C[:c,:d])[:d,:a]
    rhs = parsetensorproduct(rhs)
    if head == :(<=) || head == :(=>)
        rhs
    else
        lhs = tonameindex(lhs)[1]
        op = Dict(:(:=) => :(=),:(=) => :(.=),:(+=) => :(.+=),:(-=) => :(.-=))[head]
        Expr(op,lhs,rhs)
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

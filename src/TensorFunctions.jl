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
function toindint(ex::Expr)
    if isindexproduct(ex)
        if typeof(ex.args[2]) == Int; ex.args[3],ex.args[2]
        else                        ; ex.args[2],ex.args[3]
        end
    else
        error("not product of index and Int")
    end
end

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
    elseif issimpletensor(ex)
        indslis = [ex.args[2:end]]
        duplicateindex(indslis)
    else
        error("ex should be product of tensors")
    end
end
#= end elementary functions for parse =#

# main steps of parse =#
function bonddimdict(ex::Expr)
    if !istensorproduct(ex)
        error("not tensorproduct")
    end
    resdict = Dict{QuoteNode,T where T <:Union{Int,Symbol,Expr,Nothing}}()
    pairedindexlis = []
    for tens in ex.args[2:end]
        pos = 1
        posorig = 1
        for ind in tens.args[2:end]
            if issymbol(ind)
                resdict[ind] = :(size($(tens.args[1]),$pos))
                pos += 1
                posorig += 1
            elseif ispairedindex(ind)
                tmp = []
                for indd in ind.args
                    if typeof(indd) == Expr
                        resdict[indd.args[2]] = indd.args[3]
                    elseif typeof(indd) == QuoteNode && !haskey(resdict,indd)
                        resdict[indd] = nothing
                    end
                    pos += 1
                    push!(tmp,indd)
                end
                push!(pairedindexlis,tmp)
                posorig += 1
            elseif isindexproduct(ind)
                indname,posshift=toindint(ind)
                resdict[indname] = :(size($(tens.args[1]))[$pos:$pos+$posshift-1]|>prod)
                pos += posshift
                posorig += posshift
            else
                error("cannot parse")
            end
        end
    end
    for i in keys(resdict)
        if resdict[i] == nothing
            tmp = []
            for j in pairedindexlis
                if i in j
                    push!(tmp,j)
                end
            end
            for j in tmp

            end
        end
    end
    resdict
end

function tosimpletensor(ex,arg::Dict{Symbol,<:Union{Int,Symbol,Expr,Nothing}})
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

order(ord) = error("not supported now")
function order(ord::NTuple{N,Symbol} where N)
    ord
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
#= end main steps of parse =#


function tensorproductmain(ex,ord)
    head,lhs,rhs = toheadlhsrhs(ex) # rhs = A[:a,:b] * B[(:b,:c|hoge)] * C[(:c,:d),:e,:e]
    arg = bonddimdict(rhs)
    rhs = tosimpletensor(rhs,arg) # A[:a,:b] * reshape(B)[:b,:c] * reshape(C)[:c,:d,:e,:e]
    rhs = taketrace(rhs) # A[:a,:b] * reshape(B)[:b,:c] * trace(reshape(C))[:c,:d]
    rhs = makepairwised(rhs,order(ord)) # (A[:a,:b] * B[:b,:c])[:a,:c] * C[:c,:d]
    rhs = Expr(:ref,rhs,tonameindex(lhs)[2]...) # ((A[:a,:b] * B[:b,:c])[:a,:c] * C[:c,:d])[:d,:a]
    rhs = parsetensorproduct(rhs)
    # reshape result here
    if !(head in [:(<=),:(=>)])
        lhs = tonameindex(lhs)[1]
        op = Dict(:(:=) => :(=),:(=) => :(.=),:(+=) => :(.+=),:(-=) => :(.-=))[head]
        rhs = Expr(op,lhs,rhs)
    end
    rhs
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

module TensorFunctions

using LinearAlgebra,TensorOperations,LinearMaps

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

issimpletensor(ex,inrhs::Bool=true) = false
issimpletensor(ex::Expr,inrhs::Bool=true) =
    (ex.head == :ref && all(ex.args[2:end] .|> issymbol)) ||
    (ex.head == :vect && all(ex.args .|> issymbol) && !inrhs)

istensorproduct(ex) = false
istensorproduct(ex::Expr) = ex.head == :call && ex.args[1] == :* &&
    all(ex.args[2:end] .|> x -> istensor(x))

issimpletensorproduct(ex) = false
issimpletensorproduct(ex::Expr) = ex.head == :call && ex.args[1] == :* &&
    all(ex.args[2:end] .|> x -> issimpletensor(x))
#= end Bool functions =#

#= elementary functions for parser =#
function toindint(ex::Expr)
    if !isindexproduct(ex)
        error("not Int*Symbos")
    end
    if typeof(ex.args[2]) == QuoteNode
        ex.args[2],ex.args[3]
    else
        ex.args[3],ex.args[2]
    end
end

function toname(ex::Expr)
    if !istensor(ex)
        error("not tensor")
    end
    ex.args[1]
end

function toindex(ex::Expr)
    if !issimpletensor(ex,true) && !issimpletensor(ex,false)
        error("not tensor")
    end
    (ex.head == :ref) ? ex.args[2:end] : ex.args
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

function duplicateindex(indslis::Array{<:Any,1})
    nodup = Any[]; dup = Any[]
    for inds in indslis
        for ind in inds
            if ind in dup
                error("same index appears more than two times")
            elseif ind in nodup
                filter!(x->x!=ind,nodup)
                push!(dup,ind)
            else
                push!(nodup,ind)
            end
        end
    end
    nodup,dup
end
#= end elementary functions for parse =#

# main steps of parse =#
function bonddimdict(ex::Expr)
    if !istensorproduct(ex)
        error("not tensorproduct")
    end
    resdict = Dict{QuoteNode,Union{Int,Symbol,Expr,Nothing}}()
    pairedinddict = Dict{Array{QuoteNode,1},Expr}()
    for t in ex.args[2:end]
        posreshaped = posoriginal = 1
        for ind in t.args[2:end]
            if issymbol(ind)
                resdict[ind] = :(size($(t.args[1]),$posoriginal))
                posreshaped += 1
                posoriginal += 1
            elseif ispairedindex(ind)
                pairedind = QuoteNode[]
                for i in ind.args
                    if typeof(i) == Expr
                        resdict[i.args[2]] = i.args[3]
                        push!(pairedind,i.args[2])
                    elseif typeof(i) == QuoteNode && !haskey(resdict,i)
                        resdict[i] = nothing
                        push!(pairedind,i)
                    end
                end
                pairedinddict[pairedind] = :(size($(t.args[1]),$posoriginal))
                posreshaped += length(ind.args)
                posoriginal += 1
            elseif isindexproduct(ind)
                indname,posshift = toindint(ind)
                resdict[indname] =
                    :(size($(t.args[1]))[$posoriginal:$posoriginal+$posshift-1]|>prod)
                posreshaped += 1
                posoriginal += posshift
            else
                error("not index,pairedindex,int*index")
            end
        end
    end
    for i in filter(x->resdict[x]==nothing,keys(resdict))
        for j in filter(x->(i in x),keys(pairedinddict))
            if count(x->resdict[x]==nothing,j) == 1
                denom =
                Expr(:call,:prod,map(x->resdict[x],filter(x->resdict[x]!=nothing,j))...)
                resdict[i] = Expr(:call,:div,pairedinddict[j],denom)
                break
            end
        end
        if resdict[i] == nothing
            error("cannot determine the dim")
        end
    end
    resdict
end

function tosimpletensor!(ex,bddict::Dict{QuoteNode,<:Any})
    if !istensorproduct(ex)
        error("not tensorproduct")
    end
    ex.args[2:end] .= map(x->tosimpletensor(x,bddict),ex.args[2:end])
end

function tosimpletensor(ex,bddict::Dict{QuoteNode,<:Any})
    if !istensor(ex)
        error("not tensor")
    end
    if issimpletensor(ex)
        return ex
    end
    tnameout = Expr(:call,:reshape,ex.args[1])
    exout = Expr(:ref,tnameout)
    for i in ex.args[2:end]
        if issymbol(i)
            push!(tnameout.args,bddict[i])
            push!(exout.args,i)
        elseif isindexproduct(i)
            push!(tnameout.args,bddict[toindint(i)[1]])
            push!(exout.args,toindint(i)[1])
        elseif ispairedindex(i)
            for j in i.args
                if typeof(j) == QuoteNode
                    push!(tnameout.args,bddict[j])
                    push!(exout.args,j)
                else
                    push!(tnameout.args,bddict[j.args[2]])
                    push!(exout.args,j.args[2])
                end
            end
        end
    end
    exout
end

function taketrace!(ex::Expr,tracefunc)
    ex.args[2:end] .= map(x->taketrace(x,tracefunc),ex.args[2:end])
end

function taketrace(ex::Expr,tracefunc)
    if !istensor(ex)
        error("not tensor")
    end
    tname,tind = toname(ex),toindex(ex)
    if length(duplicateindex([tind])[2]) == 0
        ex
    else
        newtind = duplicateindex([tind])[1]
        Expr(:ref,:($tracefunc($tname,$tind,$newtind)),newtind...)
    end
end

function makepairwised(ex::Expr,ord::NTuple{N,QuoteNode} where N)
    if !issimpletensorproduct(ex)
        error("input is not tensor producr")
    end
    if length(ex.args) == 3
        return ex
    end
    indslis = [i.args[2:end] for i in ex.args[2:end]]
    nodup,dup = duplicateindex(indslis)
    contracttensor = Int[]
    if length(dup) == 0
        push!(contracttensor,2,3)
    else
        contractind = nothing
        for i in ord
            if i in dup
                contractind = i
                break
            end
        end
        if contractind != nothing
            append!(contracttensor,
                filter(x->(contractind in ex.args[x].args[2:end]),2:length(ex.args)))
        else
            push!(contracttensor,2,3)
        end
    end
    newind,dst = duplicateindex([i.args[2:end] for i in ex.args[contracttensor]])
    exout = Expr(:call,:*,
        map(i->ex.args[i],filter(x->!(x in contracttensor),2:length(ex.args)))...)
    push!(exout.args,Expr(:ref,Expr(:call,:*,ex.args[contracttensor]...),newind...))
    makepairwised(exout,ord)
end

function parsetensorproduct(ex,contractor)
    if !istensor(ex)
        error("ex should be tensor")
    elseif istensorproduct(ex.args[1])
        l = parsetensorproduct(ex.args[1].args[2],contractor)
        r = parsetensorproduct(ex.args[1].args[3],contractor)
        return :($contractor($l,
            $(ex.args[1].args[2].args[2:end]|>Tuple),
            $r,
            $(ex.args[1].args[3].args[2:end]|>Tuple),
            $(ex.args[2:end]|>Tuple)))
    else
        :($(ex.args[1]))
    end
end
#= end main steps of parse =#

#= main routine =#
function tensorproductmain(ex,ord;order=x->x,tracefunc=tensortrace,contractor=tensorcontract)
    head,lhs,rhs = toheadlhsrhs(ex)
    bdims = bonddimdict(rhs)
    tosimpletensor!(rhs,bdims)
    taketrace!(rhs,tracefunc)
    rhs = makepairwised(rhs,ord|>order)
    lhsind,lhsreshape = toindreshape(lhs|>toindex,bdims)
    rhs = Expr(:ref,rhs,lhsind...)
    rhs = parsetensorproduct(rhs,contractor)
    if lhsreshape != nothing
        rhs = Expr(:call,:reshape,rhs,lhsreshape...)
    end
    if !(head in [:(<=),:(=>)])
        lhs = toname(lhs)
        op = Dict(:(:=) => :(=),:(=) => :(.=),:(+=) => :(.+=),:(-=) => :(.-=))[head]
        rhs = Expr(op,lhs,rhs)
    end
    rhs
end

macro tensorfunc(ex::Expr)
    esc(tensorproductmain(ex))
end

function tensormapmain(ex::Expr)
    ex
end

macro tensormap(ex::Expr)
    esc(tensormapmain(ex))
end
#= end main routine =#

end # module

#= Bool functions =#
isqnode(ex) = false
isqnode(ex::QuoteNode) = true
isqnodewithdim(ex) = false
isqnodewithdim(ex::Expr) = ex.head == :call && length(ex.args) == 3 &&
    ex.args[1] == :| && isqnode(ex.args[2])
isintindexproduct(ex) = false
isintindexproduct(ex::Expr) = ex.head == :call && length(ex.args) == 3 &&
    ex.args[1] == :* && ex.args[2:3] .|> typeof |> Set == (Int,QuoteNode) |> Set

isrhspairedindex(ex) = false
isrhspairedindex(ex::Expr) = ex.head == :tuple &&
    all(ex.args .|> x -> isqnodewithdim(x) || isqnode(x))
isrhsindex(ex) = isqnode(ex) || isrhspairedindex(ex) || isintindexproduct(ex)
isrhstensor(ex) = false
isrhstensor(ex::Expr) = ex.head == :ref && all(ex.args[2:end] .|> isrhsindex)
isrhssimpletensor(ex) = false
isrhssimpletensor(ex::Expr) = ex.head == :ref && all(ex.args[2:end] .|> isqnode)

islhspairedindex(ex) = false
islhspairedindex(ex::Expr) = ex.head == :tuple && all(ex.args .|> isqnode)
islhsindex(ex) = isqnode(ex) || islhspairedindex(ex)
islhstensor(ex) = false
islhstensor(ex::Expr) = ex.head == :vect && all(ex.args .|> islhsindex)
islhssimpletensor(ex) = false
islhssimpletensor(ex::Expr) =  ex.head == :vect && all(ex.args .|> isqnode)

istensorproduct(ex) = false
istensorproduct(ex::Expr) = ex.head == :call && ex.args[1] == :* &&
    all(ex.args[2:end] .|> isrhstensor)
issimpletensorproduct(ex) = false
issimpletensorproduct(ex::Expr) = ex.head == :call && ex.args[1] == :* &&
    all(ex.args[2:end] .|> isrhssimpletensor)
#= end Bool functions =#

#= elementary functions for parser =#
function toindint(ex::Expr)
    if typeof(ex.args[2]) == QuoteNode
        ex.args[2],ex.args[3]
    else
        ex.args[3],ex.args[2]
    end
end

function todupind(indslis::Array{Array{QuoteNode,1},1})
    nodup = QuoteNode[]
    dup = QuoteNode[]
    for inds in indslis
        for i in inds
            if i in dup
                error(i," appears theree times")
            elseif i in nodup
                filter!(x->x!=i,nodup)
                push!(dup,i)
            else
                push!(nodup,i)
            end
        end
    end
    nodup,dup
end
todupind(indslis::Array{QuoteNode,1}) = todupind([indslis])
#= end elementary functions for parse =#

# main steps of parse =#
function headlhsrhs(ex::Expr)
    if ex.head == :call && ex.args[1] == :(<=) && length(ex.args) == 3 &&
        islhstensor(ex.args[2]) && istensorproduct(ex.args[3])
        ex.args[1],ex.args[2],ex.args[3]
    elseif ex.head == :call && ex.args[1] == :(=>) && length(ex.args) == 3 &&
        islhstensor(ex.args[3]) && istensorproduct(ex.args[2])
        ex.args[1],ex.args[3],ex.args[2]
    else
        error("cannot parse the Expr")
    end
end

function bonddimdict(ex::Expr)
    resdict = Dict{QuoteNode,Union{Int,Symbol,Expr,Nothing}}()
    pairedinddict = Dict{Array{QuoteNode,1},Expr}()
    for t in ex.args[2:end]
        posreshaped = posoriginal = 1
        for ind in t.args[2:end]
            if isqnode(ind)
                resdict[ind] = :(size($(t.args[1]),$posoriginal))
                posreshaped += 1
                posoriginal += 1
            elseif isrhspairedindex(ind)
                pairedind = QuoteNode[]
                for i in ind.args
                    if typeof(i) == Expr
                        resdict[i.args[2]] = i.args[3]
                        push!(pairedind,i.args[2])
                    elseif typeof(i) == QuoteNode
                        if !haskey(resdict,i)
                            resdict[i] = nothing
                        end
                        push!(pairedind,i)
                    end
                end
                pairedinddict[pairedind] = :(size($(t.args[1]),$posoriginal))
                posreshaped += length(ind.args)
                posoriginal += 1
            elseif isintindexproduct(ind)
                indname,posshift = toindint(ind)
                resdict[indname] =
                    :(size($(t.args[1]))[$posoriginal:$posoriginal+$posshift-1]|>prod)
                posreshaped += 1
                posoriginal += posshift
            end
        end
    end
    for i in filter(x->resdict[x]==nothing,keys(resdict))
        for j in filter(x->(i in x),keys(pairedinddict))
            if count(x->resdict[x]==nothing,j) == 1
                denom =
                Expr(:call,:*,map(x->resdict[x],filter(x->resdict[x]!=nothing,j))...)
                resdict[i] = Expr(:call,:div,pairedinddict[j],denom)
                break
            end
            error("cannot predict the dimension of ",i)
        end
    end
    resdict
end

function tosimpletensor!(ex,bddict::Dict{QuoteNode,<:Any})
    ex.args[2:end] .= map(x->_tosimpletensor(x,bddict),ex.args[2:end])
end
function _tosimpletensor(ex,bddict::Dict{QuoteNode,<:Any})
    if isrhssimpletensor(ex)
        return ex
    end
    tname = Expr(:call,:reshape,ex.args[1])
    exout = Expr(:ref,tname)
    for i in ex.args[2:end]
        if isqnode(i)
            push!(tname.args,bddict[i])
            push!(exout.args,i)
        elseif isintindexproduct(i)
            push!(tname.args,bddict[toindint(i)[1]])
            push!(exout.args,toindint(i)[1])
        elseif isrhspairedindex(i)
            for j in i.args
                if typeof(j) == QuoteNode
                    push!(tname.args,bddict[j])
                    push!(exout.args,j)
                else
                    push!(tname.args,bddict[j.args[2]])
                    push!(exout.args,j.args[2])
                end
            end
        end
    end
    exout
end

function taketrace!(ex::Expr)
    ex.args[2:end] .= map(_taketrace,ex.args[2:end])
end
function _taketrace(ex::Expr)
    tname,tind = ex.args[1],ex.args[2:end]
    if length(todupind(tind)[2]) == 0
        ex
    else
        newtind = todupind(tind)[1]
        Expr(:ref,:($tracefunc($tname,$tind,$newtind)),newtind...)
    end
end

function makepairwised(ex::Expr,ord::Array{QuoteNode,1})
    if length(ex.args) == 3
        return ex
    end
    nodup,dup = todupind([QuoteNode.(i.args[2:end]) for i in ex.args[2:end]])
    contracttensor = Int[]
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
    newind,dst = todupind([QuoteNode(i.args[2:end]) for i in ex.args[contracttensor]])
    exout = Expr(:call,:*,
        map(i->ex.args[i],filter(x->!(x in contracttensor),2:length(ex.args)))...)
    push!(exout.args,Expr(:ref,Expr(:call,:*,ex.args[contracttensor]...),newind...))
    makepairwised(exout,ord)
end

function parsetensorproduct(ex::Expr)
    if istensorproduct(ex.args[1])
        l = parsetensorproduct(ex.args[1].args[2])
        r = parsetensorproduct(ex.args[1].args[3])
        :($contractfunc($l,$(ex.args[1].args[2].args[2:end]|>Tuple),
            $r,$(ex.args[1].args[3].args[2:end]|>Tuple),$(ex.args[2:end]|>Tuple)))
    else
        :($(ex.args[1]))
    end
end

function toindreshape(indslis,bddict::Dict{QuoteNode,<:Any})
    resultindex = []
    resultshape = []
    for i in indslis
        if isqnode(i)
            push!(resultindex,i)
            push!(resultshape,bddict[i])
        else
            append!(resultindex,i.args)
            push!(resultshape,Expr(:call,:*,map(x->bddict[x],i.args)...))
        end
    end
    resultindex,resultshape
end
#= end main steps of parse =#

#= main routine =#
function tensorproductmain(ex::Expr,ord::Array{QuoteNode,1})
    head,lhs,rhs = headlhsrhs(ex)
    bdims = bonddimdict(rhs)
    tosimpletensor!(rhs,bdims)
    taketrace!(rhs,tracefunc)
    rhs = makepairwised(rhs,ord)
    lhsind,lhsreshape = toindreshape(lhs.args,bdims)
    rhs = Expr(:ref,rhs,lhsind...)
    rhs = parsetensorproduct(rhs)
    if length(lhs.args) != length(lhsind)
        rhs = Expr(:call,:reshape,rhs,lhsreshape...)
    end
    rhs
end

macro tensorfunc(ord::Expr,ex::Expr)
    esc(tensorproductmain(ex,order(ord)))
end

macro tensorfunc(ex::Expr)
    esc(tensorproductmain(ex,order()))
end
#= end main routine =#

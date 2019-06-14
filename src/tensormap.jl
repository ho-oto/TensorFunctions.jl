function tensormapmain(ex,ord;order=order,ishermitian=false,tracefunc=tensortrace,contractor=tensorcontract)
    head,lhs,rhs = toheadlhsrhs(ex)
    bdims = bonddimdict(rhs)
    if lhs |> toindex |> length != 2
        error("cannot convert to map")
    end
    mapindsl,mapindsr = lhs |> toindex
    if typeof(mapindsr) != QuoteNode
        mapdimr = Expr(:*,map(x->bdims[x],mapindsr.args)...)
    else
        mapdimr = bdims[mapindsr]
    end
    if typeof(mapindsl) != QuoteNode
        mapdiml = Expr(:*,map(x->bdims[x],mapindsl.args)...)
    else
        mapdiml = bdims[mapindsl]
    end
    tnamefromr,tnamefroml = gensym(),gensym()
    tfromr = Expr(:ref,tnamefromr,mapindsr)
    tfroml = Expr(:ref,tnamefroml,mapindsl)
    fnamefromr,fnamefroml = gensym(),gensym()
    exfromr,exfroml = copy(rhs),copy(rhs)
    push!(exfromr.args,tfromr)
    push!(exfroml.args,tfroml)
    typeofmap = Expr(:call,promote_type,
        Expr(:...,
            Expr(:.,:eltype,
                Expr(:tuple,
                    Expr(:vect,toname.(rhs.args[2:end])...)
                )
            )
        )
    )
    exfromr,exfroml =
        Expr(:call,:(<=),Expr(:vect,mapindsl),exfromr),Expr(:call,:(<=),Expr(:vect,mapindsl),exfroml)
    funcfromr =
        tensorproductmain(exfromr,ord,order=order,tracefunc=tracefunc,contractor=contractor)
    funcfroml =
        tensorproductmain(exfroml,ord,order=order,tracefunc=tracefunc,contractor=contractor)
    if ishermitian
        return quote
            $fnamefromr($tnamefromr) = $funcfromr
            LinearMap{$typeofmap}($fnamefromr,$mapdimr,ishermitian=true)
        end
    else
        return quote
            $fnamefromr($tnamefromr) = $funcfromr
            $fnamefroml($tnamefroml) = $funcfroml
            LinearMap{$typeofmap}($fnamefromr,$fnamefroml,$mapdimr,$mapdiml)
        end
    end
end

macro tensormap(ord::Expr,ex::Expr)
    esc(tensormapmain(ex,ord))
end

macro tensormap(ex::Expr)
    esc(tensormapmain(ex,:((nothing,))))
end

macro tensorhmap(ord::Expr,ex::Expr)
    esc(tensormapmain(ex,ord,ishermitian=true))
end

macro tensorhmap(ex::Expr)
    dummy = Expr(:tuple,:nothing)
    esc(tensormapmain(ex,dummy,ishermitian=true))
end

function tensormapmain(ex,ord;order=order,ishermitian=false,tracefunc=tensortrace,contractor=tensorcontract)
    head,lhs,rhs = toheadlhsrhs(ex)
    bdims = bonddimdict(rhs)
    if lhs |> toindex |> length != 2
        error("cannot convert to map")
    end
    mapindsl,mapindsr = lhs |> toindex
    if mapindsr.head == :tuple
        mapdimr = Expr(:*,map(x->bdims[x],mapindsr.args)...)
    else
        mapdimr = bdims[mapindsr]
    end
    if mapindsl.head == :tuple
        mapdiml = Expr(:*,map(x->bdims[x],mapindsl.args)...)
    else
        mapdiml = bdims[mapindsl]
    end
    tnamefromr,tnamefroml = gensym(),gensym()
    tfromr = Expr(:ref,tnamefromr,mapindsr)
    tfroml = Expr(:ref,tnamefroml,mapindsl)
    fnamefromr,fnamefroml = gensym(),gensym()
    exfromr,exfroml = copy(rhs),copy(rhs)
    push!(exfromr,tfromr)
    push!(exfroml,tfroml)
    typeofmap = Expr(:call,promote_type,
        Expr(:.,:eltype,
            Expr(:tuple,
                Expr(:vect,toname.(rhs.args[2:end])...)
            )
        )
    )
    funcfroml =
        tensorproductmain(exfroml,ord,order=order,tracefunc=tracefunc,contractor=contractor)
    funcfromr =
        tensorproductmain(exfromr,ord,order=order,tracefunc=tracefunc,contractor=contractor)
    if ishermitian
        return quote
            $fnamefroml($tnamefroml) = $funcfroml
            LinearMap{$typeofmap}($fnamefromr,$fnamefroml,$mapdiml,ishermitian=true)
        end
    else
        return quote
            $fnamefroml($tnamefroml) = $funcfroml
            $fnamefromr($tnamefromr) = $funcfromr
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

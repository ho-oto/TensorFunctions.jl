function tensormapmain(ex,ord;order=order,tracefunc=tensortrace,contractor=tensorcontract)
    head,lhs,rhs = toheadlhsrhs(ex)
    if lhs|>toindex|>length != 2
        error("cannot convert to map")
    end
    maplinds,maprinds = lhs|>toindex
    t1 = gensym()
    t2 = gensym()
    fromr = Expr(:ref,t1,maprinds)
    froml = Expr(:ref,t2,maplinds)

    tmptmp1 = gensym()
    tmptmp2 = gensym()

    tmp1 = tensorproductmain(ex,ord,order=order,tracefunc=tracefunc,contractor=contractor)
    tmp2 = tensorproductmain(ex,ord,order=order,tracefunc=tracefunc,contractor=contractor)
    quote
        $tmptmp1($t1) = $tmp1
        $tmptmp2($t2) = $tmp2
        LinearMap{}($tmptmp1,$tmptmp2)
    end
end

macro tensormap(ord::Expr,ex::Expr)
    esc(tensormapmain(ex,ord))
end

macro tensormap(ex::Expr)
    esc(tensormapmain(ex,:((nothing,))))
end

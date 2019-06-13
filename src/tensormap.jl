


macro tensormap(ord::Expr,ex::Expr)
    esc(tensormapmain(ex,ord))
end

macro tensormap(ex::Expr)
    esc(tensormapmain(ex,:((nothing,))))
end

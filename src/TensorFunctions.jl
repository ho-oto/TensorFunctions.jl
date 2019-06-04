module TensorFunctions

using TensorOperations
using LinearMaps
using TensorCast

export @tensorfunc, @tensormap

macro tensorfunc(ex::Expr)
    ex
end

macro tensormap(ex::Expr)
    ex
end

end # module

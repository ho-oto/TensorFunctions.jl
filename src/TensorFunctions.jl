module TensorFunctions

using LinearAlgebra,TensorOperations#,LinearMaps

export @tensorfunc#,@tensormap

include("tensorfunc.jl")
#include("tensormap.jl")

end # module

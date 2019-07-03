using Documenter
using TensorFunctions

makedocs(
    sitename = "TensorFunctions",
    format = Documenter.HTML(),
    modules = [TensorFunctions]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

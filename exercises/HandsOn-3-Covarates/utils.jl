"""
    unroll(nt::NamedTuple) -> NamedTuple

Transforms a NamedTuple by unrolling its array fields into individual key–value pairs. 
For each array field, each element is assigned a new key formed by appending a 
sequentially increasing subscript to the original field name. Non-array fields are 
kept unchanged.

# Examples
```julia
nt = (a = [10, 20], b = 5)
# unroll(nt) returns a NamedTuple with keys like :a₁, :a₂, and :b, for example:
# (:a₁ => 10, :a₂ => 20, :b => 5)
```
"""
function unroll(nt::NamedTuple)
    mapreduce(_unroll, merge, pairs(nt))
end

unroll(f::Function; kwargs...) = x -> unroll(f, x; kwargs...)

_unroll(p::Pair) = _unroll(p.first, p.second)
_unroll(key::Symbol, val::Any) = (; key => val)
function _unroll(key::Symbol, val::AbstractArray{T}) where {T}
    inds = CartesianIndices(val)
    syms = Vector{Symbol}(undef, length(inds))
    flat_vals = Vector{T}(undef, length(inds))
    for (i, ind) ∈ enumerate(inds)
        syms[i] = Symbol(string(key, join(map(Pumas._to_subscript, Tuple(ind)), ",")))
        flat_vals[i] = val[ind]
    end
    NamedTuple{Tuple(syms)}(flat_vals)
end

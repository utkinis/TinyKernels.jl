import MacroTools: splitdef, combinedef

@inline __validindex(ndrange) = __get_index() <= length(ndrange)

macro tiny(expr)
    def = splitdef(expr)
    # create GPU kernel
    pushfirst!(def[:args], :__ndrange__)
    def[:body] = quote
        if !TinyKernels.__validindex(__ndrange__)
            return
        end
        $(def[:body])
    end
    return esc(combinedef(def))
end

macro indices()        esc(:( @inbounds         Tuple( __ndrange__[TinyKernels.__get_index()]) )) end
macro linearindex()    esc(:( @inbounds LinearIndices(__ndrange__)[TinyKernels.__get_index()]  )) end
macro cartesianindex() esc(:( @inbounds                __ndrange__[TinyKernels.__get_index()]  )) end
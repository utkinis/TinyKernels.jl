import MacroTools: splitdef, combinedef

macro tiny(expr)
    def = splitdef(expr)
    # create GPU kernel
    pushfirst!(def[:args], :__ndrange__)
    def[:body] = quote
        if TinyKernels.__validindex(__ndrange__)
            $(def[:body])
        end
    end
    return esc(combinedef(def))
end

macro indices() esc(:(TinyKernels.__get_indices(Val(ndims(__ndrange__))))) end

import MacroTools: splitdef, combinedef

macro tiny(expr)
    def = splitdef(expr)
    # create GPU kernel
    pushfirst!(def[:args], :__ndrange__)
    def[:body] = quote
        $__INDEX__ = TinyKernels.__get_index()
        if !($__INDEX__ <= length(__ndrange__))
            return
        end
        $(def[:body])
    end
    return esc(combinedef(def))
end

macro indices()        esc(:( @inbounds         Tuple( __ndrange__[$__INDEX__]) )) end
macro linearindex()    esc(:( @inbounds LinearIndices(__ndrange__)[$__INDEX__]  )) end
macro cartesianindex() esc(:( @inbounds                __ndrange__[$__INDEX__]  )) end
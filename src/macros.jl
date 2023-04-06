import MacroTools: splitdef, combinedef, postwalk, @capture

function _replace_returns(body)
    goto_ex = Meta.parse("@goto early_exit")
    postwalk(body) do ex
        @capture(ex, return x_) ? goto_ex : ex
    end
end

function make_gpu!(def)
    def[:body] = quote
        $__INDEX__ = TinyKernels.__get_index()
        if !($__INDEX__ <= length(__ndrange__))
            return
        end
        $(def[:body])
        return
    end
    return
end

function make_cpu!(def)
    _replace_returns(def[:body])
    def[:body] = quote
        for $__INDEX__ in eachindex(__ndrange__)
            $(def[:body])
            @label early_exit
        end
        return
    end
    return
end

macro tiny(expr)
    def = splitdef(expr)
    name = def[:name]

    pushfirst!(def[:args], :__ndrange__)

    # check for and remove final "return"
    if def[:body].args[end] == :(return)
        deleteat!(def[:body].args, lastindex(def[:body].args))
    end

    def_gpu = deepcopy(def)
    def_cpu = deepcopy(def)

    def_gpu[:name] = Symbol(:gpu_, def_gpu[:name])
    def_cpu[:name] = Symbol(:cpu_, def_cpu[:name])

    # create GPU and CPU kernels
    make_gpu!(def_gpu)
    make_cpu!(def_cpu)

    # create constructors
    constructors = quote
        if $(name isa Symbol ? :(!@isdefined($name)) : true)
            function $name(device::AbstractGPUDevice)
                return Kernel($(def_gpu[:name]), device)
            end
            function $name(device::CPUDevice)
                return Kernel($(def_cpu[:name]), device)
            end
        end
    end

    return Expr(:block, esc(combinedef(def_gpu)), esc(combinedef(def_cpu)), esc(constructors))
end

macro indices()        esc(:(@inbounds          Tuple(__ndrange__[$__INDEX__]))) end
macro linearindex()    esc(:(@inbounds LinearIndices(__ndrange__)[$__INDEX__]) ) end
macro cartesianindex() esc(:(@inbounds                __ndrange__[$__INDEX__]) ) end
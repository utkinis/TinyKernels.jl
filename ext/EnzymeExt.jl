module EnzymeExt

@static isdefined(Base, :get_extension) ? (import Enzyme) : (import ..Enzyme)

import TinyKernels: AbstractGPUDevice, CPUDevice, Kernel

function Enzyme.autodiff(kernel::Kernel{<:AbstractGPUDevice, Fun}) where Fun
    fun = kernel.fun
    function df(ctx, args...)
        Enzyme.autodiff_deferred(Enzyme.Reverse, fun::Fun, Enzyme.Const, ctx, args...)
        return
    end
    similar(kernel, df)
end

function Enzyme.autodiff(kernel::Kernel{CPUDevice, Fun}) where Fun
    fun = kernel.fun
    function df(ctx, args...)
        Enzyme.autodiff(Enzyme.Reverse, fun::Fun, Enzyme.Const, ctx, args...)
        return
    end
    similar(kernel, df)
end

end # module
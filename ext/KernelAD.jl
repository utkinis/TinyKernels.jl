module KernelAD

@static isdefined(Base, :get_extension) ? (import Enzyme) : (import ..Enzyme)

import TinyKernels: GPUDevice, CPUDevice, Kernel

function Enzyme.autodiff(kernel::Kernel{<:GPUDevice, Fun}) where Fun
    fun = kernel.fun
    function df(ctx, args...)
        Enzyme.autodiff_deferred(fun::Fun, Enzyme.Const, ctx, args...)
        return nothing
    end
    similar(kernel, df)
end

function Enzyme.autodiff(kernel::Kernel{<:CPUDevice, Fun}) where Fun
    fun = kernel.fun
    function df(ctx, args...)
        Enzyme.autodiff(Enzyme.Reverse, fun::Fun, Enzyme.Const, ctx, args...)
        return nothing
    end
    similar(kernel, df)
end

end # module
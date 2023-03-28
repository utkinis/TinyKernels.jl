module KernelAD

import Enzyme

import TinyKernels: Kernel, GPUDevice
import TinyKernels.CPUBackend: CPUDevice

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
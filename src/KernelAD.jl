module KernelAD

import Enzyme
import TinyKernels: Kernel

function Enzyme.autodiff(kernel::Kernel{<:Any, Fun}) where Fun
    fun = kernel.fun
    function df(ctx, args...)
        Enzyme.autodiff_deferred(fun::Fun, Enzyme.Const, ctx, args...)
        return nothing
    end
    similar(kernel, df)
end

end # module
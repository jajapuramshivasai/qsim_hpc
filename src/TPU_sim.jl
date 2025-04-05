using XLA

function matmul_kernel(x_ref, y_ref, z_ref, acc_ref; nsteps, transpose_rhs)
  # Kernel implementation
end

function matmul(x, y; bm=128, bk=128, bn=128, transpose_rhs=false)
  # Matrix multiplication implementation using pallas_call
end

result = matmul(x, y, bm=512, bk=1024, bn=1024, transpose_rhs=true)

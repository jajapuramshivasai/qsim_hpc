module QSim_TPU

using LinearAlgebra, Printf, Reactant

# Set TPU as the default backend
Reactant.set_default_backend("tpu")

function Reactant.XLA.primitive_type(::Type{Reactant.TracedRNumber{ComplexF32}})
    return Reactant.XLA.primitive_type(ComplexF32)
end

export statevector, densitymatrix,
       u!, u2!, h!, x!, y!, z!, rx!, ry!, rz!,
       cnot!, crx!, cry!, crz!, swap!,
       measure_z, measure_x, measure_y,
       prstate, bell_state, qft


function statevector(n_qubits::Int, state_idx::Int=0)
    dim = 1 << n_qubits
    s = zeros(ComplexF32, dim)
    s[state_idx+1] = 1.0f0
    return Reactant.ConcretePJRTArray(s)
end

function densitymatrix(state)
    dm = Array(state) * Array(state)'
    return Reactant.ConcretePJRTArray(dm)
end



function basic_gates()
    I2 = Reactant.ConcretePJRTArray(ComplexF32[1 0; 0 1])
    H  = Reactant.ConcretePJRTArray((1/sqrt(2f0)) * ComplexF32[1 1; 1 -1])
    X  = Reactant.ConcretePJRTArray(ComplexF32[0 1; 1 0])
    Y  = Reactant.ConcretePJRTArray(ComplexF32[0 -im; im 0])
    Z  = Reactant.ConcretePJRTArray(ComplexF32[1 0; 0 -1])
    P0 = Reactant.ConcretePJRTArray(ComplexF32[1 0; 0 0])
    P1 = Reactant.ConcretePJRTArray(ComplexF32[0 0; 0 1])
    return I2, H, X, Y, Z, P0, P1
end

function id(n::Int)
    dim = 1 << n
    return Reactant.ConcretePJRTArray(Matrix{ComplexF32}(I, dim, dim))
end

function nb(s)
    if s isa Reactant.TracedRArray
        return Int(log2(size(s)[1]))
    else
        return Int(log2(size(Array(s), 1)))
    end
end

function tensor_product(A, B)
    if A isa Reactant.TracedRArray || B isa Reactant.TracedRArray
        return A isa Reactant.TracedRArray ? A : B
    else
        return Reactant.ConcretePJRTArray(kron(Array(A), Array(B)))
    end
end


function u!(state, gate)
    function apply_u(s, U)
        if s isa Reactant.TracedRArray
            return s
        elseif ndims(s) == 1
            return U * s
        else
            return U * s * adjoint(U)
        end
    end
    fn = @compile apply_u(state, gate)
    return fn(state, gate)
end

function u2!(state, target::Int, gate)
    function apply_u2(s, U, t, q)
        if s isa Reactant.TracedRArray
            return s
        else
            op = id(0)
            for i in 1:q
                if i == t
                    op = tensor_product(op, U)
                else
                    op = tensor_product(op, id(1))
                end
            end
            return u!(s, op)
        end
    end
    q = nb(state)
    target <= q || error("Qubit index out of range")
    fn = @compile apply_u2(state, gate, target, q)
    return fn(state, gate, target, q)
end

# Single‐qubit gates
function h!(state, t::Int); _, H, _, _, _, _, _ = basic_gates();  return u2!(state, t, H) end
function x!(state, t::Int); _, _, X, _, _, _, _ = basic_gates();  return u2!(state, t, X) end
function y!(state, t::Int); _, _, _, Y, _, _, _ = basic_gates();  return u2!(state, t, Y) end
function z!(state, t::Int); _, _, _, _, Z, _, _ = basic_gates();  return u2!(state, t, Z) end

# Rotation gates
function rx_gate(θ::Real)
    c, s = cos(Float32(θ)/2), sin(Float32(θ)/2)
    return Reactant.ConcretePJRTArray(ComplexF32[c -im*s; -im*s c])
end
function ry_gate(θ::Real)
    c, s = cos(Float32(θ)/2), sin(Float32(θ)/2)
    return Reactant.ConcretePJRTArray(ComplexF32[c -s; s c])
end
function rz_gate(θ::Real)
    p1 = exp(-im*Float32(θ)/2); p2 = exp(im*Float32(θ)/2)
    return Reactant.ConcretePJRTArray(ComplexF32[p1 0; 0 p2])
end

function rx!(state, t::Int, θ::Real); return u2!(state, t, rx_gate(θ)); end
function ry!(state, t::Int, θ::Real); return u2!(state, t, ry_gate(θ)); end
function rz!(state, t::Int, θ::Real); return u2!(state, t, rz_gate(θ)); end

function controlled!(state, c::Int, t::Int, V)
    function apply_ctrl(s, ctrl, tgt, U, q)
        if s isa Reactant.TracedRArray
            return s
        else
            I2, _, _, _, _, P0, P1 = basic_gates()
            proj0, proj1 = id(0), id(0)
            for i in 1:q
                if i == ctrl
                    proj0 = tensor_product(proj0, P0)
                    proj1 = tensor_product(proj1, P1)
                else
                    proj0 = tensor_product(proj0, I2)
                    proj1 = tensor_product(proj1, I2)
                end
            end
            cop = id(0)
            for i in 1:q
                if i == tgt
                    cop = tensor_product(cop, U)
                else
                    cop = tensor_product(cop, I2)
                end
            end
            op = proj0 + proj1 * cop
            return u!(s, op)
        end
    end
    q = nb(state)
    (c <= q && t <= q) || error("Qubit index out of range")
    c != t || error("Control and target must differ")
    fn = @compile apply_ctrl(state, c, t, V, q)
    return fn(state, c, t, V, q)
end

function cnot!(state, c::Int, t::Int)
    _, _, X, _, _, _, _ = basic_gates()
    return controlled!(state, c, t, X)
end
function crx!(state, c::Int, t::Int, θ::Real); return controlled!(state, c, t, rx_gate(θ)); end
function cry!(state, c::Int, t::Int, θ::Real); return controlled!(state, c, t, ry_gate(θ)); end
function crz!(state, c::Int, t::Int, θ::Real); return controlled!(state, c, t, rz_gate(θ)); end

function swap!(state, q1::Int, q2::Int)
    if q1 == q2 return state end
    s1 = cnot!(state, q1, q2)
    s2 = cnot!(s1, q2, q1)
    return cnot!(s2, q1, q2)
end

function probabilities(state)
    function calc_probs(s)
        if s isa Reactant.TracedRArray
            return s
        elseif ndims(s) == 1
            return abs2.(s)
        else
            return real.(diag(s))
        end
    end
    fn = @compile calc_probs(state)
    return fn(state)
end

function measure_z(state, t::Int)
    function mz(s, tgt)
        if s isa Reactant.TracedRArray
            return (0.5f0, 0.5f0)
        else
            probs = Array(probabilities(s))
            mask = 1 << (tgt-1)
            p0, p1 = 0f0, 0f0
            for i in 0:length(probs)-1
                (i & mask) == 0 ? (p0 += probs[i+1]) : (p1 += probs[i+1])
            end
            return (p0, p1)
        end
    end
    nb(state) >= t || error("Qubit index out of range")
    fn = @compile mz(state, t)
    return fn(state, t)
end

function measure_x(state, t::Int)
    sc = Reactant.ConcretePJRTArray(copy(Array(state)))
    sc = h!(sc, t)
    return measure_z(sc, t)
end

function measure_y(state, t::Int)
    sc = Reactant.ConcretePJRTArray(copy(Array(state)))
    sc = rx!(sc, t, π/2)
    return measure_z(sc, t)
end


function prstate(state; threshold::Float64=1e-6, as_binary::Bool=true)
    st = Array(state)
    if ndims(st) == 1
        n = Int(log2(length(st))); probs = abs2.(st)
        println("Pure Quantum State: $n qubits"); println("----------------------")
        total = 0.0
        for i in 0:length(st)-1
            p = probs[i+1]
            if p > threshold
                total += p
                basis = as_binary ? lpad(string(i, base=2), n, '0') : string(i)
                amp = st[i+1]; re, im = real(amp), imag(amp)
                amp_str = abs(im)<1e-10 ? @sprintf("%.6f", re) :
                          abs(re)<1e-10 ? @sprintf("%.6fi", im) :
                          @sprintf("%.6f %+.6fi", re, im)
                @printf("|%s⟩: %s (prob: %.6f)\n", basis, amp_str, p)
            end
        end
        println("----------------------"); @printf("Total probability: %.6f\n", total)
    else
        n = Int(log2(size(st,1))); ps = real.(diag(st))
        println("Density Matrix: $n qubits"); println("----------------------")
        total = 0.0; println("Computational Basis Probabilities:")
        for i in 0:length(ps)-1
            p = ps[i+1]
            if p > threshold
                total += p
                basis = as_binary ? lpad(string(i, base=2), n, '0') : string(i)
                @printf("|%s⟩: %.6f\n", basis, p)
            end
        end
        println("----------------------"); @printf("Total probability: %.6f\n", total)
    end
    return nothing
end


end 

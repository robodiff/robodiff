import numpy as np

import robodiff_startup as dipv
import taichi as ti

import IPython
import time

real = ti.f64
ti_init_kwargs = {
    'default_fp':real,
    'flatten_if':True,
}

if dipv.args.cpu:
    if dipv.args.cpu_max_threads > 0:
        ti.init(arch=ti.cpu,  cpu_max_num_threads=dipv.args.cpu_max_threads, **ti_init_kwargs)
    else:
        ti.init( arch=ti.cpu, **ti_init_kwargs)
else:
    ti.init(arch=ti.cuda, **ti_init_kwargs)

x_frozen_np = None
dim = 2

n_grid = 128
nx_grid = n_grid * (2 if dipv.args.resim or dipv.args.wide_world else 1)

dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
nu = 0.25
max_steps = dipv.args.max_steps

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

particle_type = ti.field(ti.i32) # constant
x, v = vec(), vec() # grads cleared. X re-loaded prior to sim. 
m = scalar() # set prior to sim.
grid_v_in, grid_m_in = vec(), scalar() # clear_grid clears grad + value. called per time step.
grid_v_out = vec() # grad cleared per time step, set in grid_op
C, F = mat(), mat() # F read from step 0 written to step 1+
act_model =  mat()
loss = scalar() # cleared at start
E_base = scalar() # set prior to sim
friction = scalar() # set prior to sim
total_mass = scalar() # cleared and set just before sim.

internal_damping = scalar() # set prior to sim
global_damping = scalar() # set prior to sim

actuation_frequency = scalar() # set prior to sim
actuation_amplitude = scalar() # set prior to sim
actuation_phase = scalar() # set prior to sim
actuation_bias = scalar() # set prior to sim
actuation_sharpness = scalar() # 1.0 => closer to sine wave. 10.0 => closer to square wave.
actuation_max_signal = scalar() # 0 => only expand, 10000 => expand and contract.

x_avg = vec() # cleared and set during sim
v_avg = vec() # cleared and set during sim
v_avg_final = vec() # cleared and set during sim

realized_actuation = scalar() # set during sim. prior to reading.
realized_pressure = scalar() # set during sim. prior to reading.
act_strength = scalar() # only read during sim
actuation_proportional_to_mass = ti.field(ti.i32)
gravity = vec() # only read during sim.

simulation_inclusion_threshold = scalar() # read for compilation.

def place():
    ti.root.dense(ti.ij, (max_steps, dipv.scene.n_particles)).place(realized_actuation, realized_pressure, m)
    ti.root.dense(ti.i, dipv.scene.n_particles).place(actuation_frequency, actuation_amplitude, actuation_phase, actuation_bias, particle_type)
    ti.root.dense(ti.l, max_steps).dense(ti.k, dipv.scene.n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, (nx_grid, n_grid )).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.dense(ti.i, max_steps).place(x_avg, v_avg, v_avg_final)
    ti.root.place(act_model, loss, simulation_inclusion_threshold, internal_damping, global_damping, E_base, total_mass, act_strength, actuation_proportional_to_mass, gravity, friction, actuation_sharpness, actuation_max_signal)

    ti.root.lazy_grad()

@ti.kernel
def clear_m_grads():
    for t, i in m:
        m.grad[t, i] = 0.0

@ti.kernel
def clear_loss():
  loss[None] = 0.0

@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad(t: ti.i32):
    # for all time steps and all particles
    for f, i in x:
        if f >= t:
            x.grad[f, i] = [0, 0]
            v.grad[f, i] = [0, 0]
            v[f, i] = [0, 0]
            C.grad[f, i] = [[0, 0], [0, 0]]
            F.grad[f, i] = [[0, 0], [0, 0]]

@ti.kernel
def clear_total_mass():
    total_mass[None] = 0.0

@ti.kernel
def compute_total_mass():
    for t, i  in m:
        if m[t, i] > 0.0:
            total_mass[None] += 1.0 

@ti.kernel
def clear_actuation_grad():
    for t, i in realized_actuation:
        realized_actuation[t, i] = 0.0
        realized_pressure[t, i] = 0.0

@ti.kernel
def avg_velocity(f: ti.i32):
    for p in range(dipv.scene.n_terrain_particles, dipv.scene.n_particles):
        v_avg[f] += (v[f, p]/float(dipv.scene.n_particles - dipv.scene.n_terrain_particles))


@ti.kernel
def p2g(f: ti.i32, curr_relative_act_strength: real, enableDamping: real):
    for p in range(dipv.scene.n_particles):
        if m[f, p] > simulation_inclusion_threshold:
            curr_e = E_base[None] * 2.0 if p < dipv.scene.n_terrain_particles else  E_base[None]

            base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32) # grid cell
            fx = x[f, p] * inv_dx - ti.cast(base, ti.i32) # location in grid cell
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] # weighed momentum sum for adding into grid cells
            new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p] # deformation gradient.
            J = (new_F).determinant() # volume of volume element.
            if particle_type[p] == 0:  # fluid
                sqrtJ = ti.sqrt(J)
                new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

            F[f + 1, p] = new_F
                        
            act = get_actuation(p, f, curr_relative_act_strength) *  act_strength[None] 

            if actuation_proportional_to_mass[None] == 1:
                act *= m[f, p]

            realized_actuation[f, p] = act
            realized_pressure[f, p] = J

            A = act_model * act

            cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            mass = m[f, p]
            if particle_type[p] == 0:
                cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * m[f, p] * curr_e
            else:
                mu = m[f, p] * curr_e / (2 * (1 + nu))
                la = m[f, p] * curr_e * nu / ((1 + nu) * (1-2 * nu))
                r, s = ti.polar_decompose(new_F)
                cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                           ti.Matrix.diag(2, la * (J - 1) * J)               

            cauchy += new_F @ A @ new_F.transpose()
            stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
            affine = stress + mass * C[f, p]

            # internal damping
            is_terrain = 1.0 * (p < dipv.scene.n_terrain_particles)
            internal_damping_multiplier = ti.max(is_terrain, ti.exp(-internal_damping*dt)) # will be 1 if is terrain, else damping amount.
            v_delta = v[f,p] - v_avg[f]
            v_delta *=  internal_damping_multiplier
            v[f,p] = v_delta + v_avg[f]

            # global damping
            v[f,p] *= ti.exp(-global_damping*dt)

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                    weight = w[i](0) * w[j](1)
                    grid_v_in[base +
                            offset] += weight * (mass * v[f, p] + affine @ dpos)
                    grid_m_in[base + offset] += weight * mass

bound = 3
top_bound = 3

@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out += dt * gravity[None]
        
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > nx_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and j > 0 and v_out[1] < 0:
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if friction[None] < 0:
                    v_out(0).val = 0
                    v_out(1).val = 0
                else:
                    lin = (v_out.transpose() @ normal)(0)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + friction[None] * lin <= 0:
                            v_out(0).val = 0
                            v_out(1).val = 0
                        else:
                            v_out = (1 + friction[None] * lin / lit) * vit
        if j == 0:
            v_out[0] = 0
            v_out[1] = 0
        if j > n_grid - top_bound and v_out[1] > 0:
            normal = ti.Vector([0.0, -1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if friction[None] < 0:
                    v_out(0).val = 0
                    v_out(1).val = 0
                else:
                    lin = (v_out.transpose() @ normal)(0)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + friction[None] * lin <= 0:
                            v_out(0).val = 0
                            v_out(1).val = 0
                        else:
                            v_out = (1 + friction[None] * lin / lit) * vit
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(dipv.scene.n_particles):
        if m[f, p] > simulation_inclusion_threshold:

            base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
            fx = x[f, p] * inv_dx - ti.cast(base, real)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector([0.0, 0.0])
            new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j]), real) - fx
                    g_v = grid_v_out[base(0) + i, base(1) + j]
                    weight = w[i](0) * w[j](1)
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

            v[f + 1, p] = new_v
            x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
            C[f + 1, p] = new_C


@ti.func
def sigmoid(x):
    tmp = ti.exp(x)
    return (tmp)/(tmp+1)

@ti.func
def get_actuation(i, t, curr_relative_act_strength):
    act = ti.sin(2 * np.pi *  actuation_frequency[i] * t * dt + actuation_phase[i])

    ret_act =  (ti.tanh(actuation_sharpness[None] * actuation_amplitude[i] * act + actuation_bias[i]))* curr_relative_act_strength
    return (ret_act < actuation_max_signal[None]) * ret_act # cap actuation signal, if actuation_max_signal set low. Used for exploring robustness of behaviors when changing actuation capabilities.
    # return (ret_act > 0) * ret_act # shrink only

@ti.kernel
def clear_x_avg():
    for s in range(0, max_steps):
        x_avg[s] = [0,0]
        v_avg_final[s] = [0,0]

@ti.kernel
def compute_x_avg(steps: ti.i32, particle_start_idx: ti.i32, particle_end_idx: ti.i32):
    for s in range(0, steps):
        for i in range(particle_start_idx, particle_end_idx):
            if m[s, i] > simulation_inclusion_threshold:
                contrib = 0.0
                if particle_type[i] == 1:
                    contrib = 1.0 / (dipv.scene.n_solid_particles - dipv.scene.n_terrain_particles)
                x_avg[s].atomic_add(contrib * x[s, i])
                v_avg_final[s].atomic_add(contrib * v[s, i])
                
@ti.kernel
def clear_v_avg(t: ti.i32):
    for s in range(t, max_steps):
        v_avg[s] = [0,0]

@ti.kernel
def compute_loss(steps_offset: ti.i32, steps: ti.i32):
    loss[None] =  x_avg[steps_offset][0] - x_avg[steps-1][0]
    
@ti.kernel
def compute_jump_loss(steps_offset: ti.i32, steps: ti.i32):
    for f in range(steps_offset, steps - 1):
        loss[None] += -1 * ti.max(v_avg_final[f][1], 0.) / steps 


@ti.complex_kernel
def advance(s, curr_relative_act_strength, enableDamping=1):
    clear_grid()
    avg_velocity(s)
    p2g(s, curr_relative_act_strength, enableDamping)
    grid_op()
    g2p(s)


@ti.complex_kernel_grad(advance)
def advance_grad(s, curr_relative_act_strength, enableDamping=1):
    clear_grid()
    avg_velocity(s)
    p2g(s, curr_relative_act_strength, enableDamping)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s, curr_relative_act_strength, enableDamping)


def pre_sim(total_steps=max_steps):
    assert total_steps <= max_steps, "Can not simulate longer than {max_steps}. Please re-run with a --max_steps cli flag that is larger."
    # simulation
    clear_loss()
    clear_total_mass()
    compute_total_mass()
    clear_v_avg(0)
    clear_particle_grad(0)

def forward(step_offset=0, total_steps=max_steps, curr_relative_act_strength=1.0, enableDamping=1, in_pre_tape_mode=False):
    half_total_steps = total_steps//2
    for s in range(step_offset, total_steps - 1): # tqdm(range(0, total_steps - 1)):
        if step_offset == 0 and in_pre_tape_mode: # beginning of simulation
            curr_relative_act_strength *= 0.0 if s < half_total_steps else (1 - s/half_total_steps)
        advance(s, curr_relative_act_strength, enableDamping)

def post_sim(step_offset=0,
                total_steps=max_steps,
                loss_mode=0,
                particle_start_idx=0,
                particle_end_idx=None):
    if particle_end_idx is None:
        particle_end_idx = dipv.scene.n_particles
    clear_x_avg()
    compute_x_avg(total_steps, particle_start_idx, particle_end_idx)
    if loss_mode == 0:
      compute_loss(step_offset, total_steps)
    elif loss_mode == 1:
      compute_jump_loss(step_offset, total_steps)
    else:
        raise ValueError("Loss Mode must be 0 or 1 (locomotion or jumping)")


def simulate(pre_grads_steps=0,
                total_steps=max_steps,
                loss_mode=0,
                enableDamping=1,
                with_grads=False,
                particle_start_idx=0,
                particle_end_idx=None,
                 **kwargs):
    clear_m_grads()
    global x_frozen_np
    if x_frozen_np is None:
        x_frozen_np = x.to_numpy()
    else:
        x.from_numpy(x_frozen_np)

    tape_step_offset = max(0, pre_grads_steps-1)

    pre_sim(total_steps=total_steps)
    if with_grads:
        forward(step_offset=0, total_steps=pre_grads_steps, curr_relative_act_strength=0.0, enableDamping=enableDamping, in_pre_tape_mode=True)
        
        with ti.Tape(loss):
            forward(step_offset=tape_step_offset, total_steps=total_steps, enableDamping=enableDamping, in_pre_tape_mode=False)
            post_sim(step_offset=tape_step_offset,
                        total_steps=total_steps,
                        loss_mode=loss_mode,
                        particle_start_idx=particle_start_idx,
                        particle_end_idx=particle_end_idx)
    else:
        forward(step_offset=0, total_steps=pre_grads_steps, curr_relative_act_strength=0.0, enableDamping=enableDamping, in_pre_tape_mode=True)
        forward(step_offset=tape_step_offset, total_steps=total_steps, enableDamping=enableDamping)
        post_sim(step_offset=pre_grads_steps,
                    total_steps=total_steps,
                    loss_mode=loss_mode,
                        particle_start_idx=particle_start_idx,
                        particle_end_idx=particle_end_idx)


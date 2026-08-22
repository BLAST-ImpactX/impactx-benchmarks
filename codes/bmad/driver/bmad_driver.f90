!+
! Program bmad_driver
!
! Standalone Bmad benchmark driver (no Tao/pytao). Reads a `beam_track_params` namelist
! (lattice file + beam_init + ran_seed + spin/n_warmup, and optional space-charge knobs), tracks a
! Gaussian beam through the lattice, times ONLY the tracking (one untimed warm-up first), and prints:
!     Track: <ns>ns
!     Validate: {json}      ! tracking: sigma_x/y, sigma_t, emit_x/y ; spin: sigma_sx/sy/sz
!
! Space charge (spacecharge scenario): when space_charge=T, one 3D OpenSpaceCharge kick (Ryne's
! integrated Green function; open-boundary Hockney FFT + CIC deposit/gather -- the SAME model as
! ImpactX/HELIX) is applied ONCE over sc_ds before the drift transport (asymmetric kick->drift, one
! field solve -- the fair single-kick convention). The native lattice fft_3d path does n_step+1 kicks
! with no endpoint halving, so it cannot do a single solve; we call OpenSpaceCharge directly instead.
!-

program bmad_driver

use bmad
use beam_mod
use open_spacecharge_mod

implicit none

type (lat_struct) lat
type (ele_struct), pointer :: ele1, ele2
type (beam_struct), target :: beam
type (beam_init_struct) beam_init

integer :: n_arg, ran_seed, n_warmup, i, sc_mesh(3)
integer(8) :: c1, c2, crate
real(rp) :: dt_ns, sc_ds, gamma
logical :: err, spin, space_charge
character(200) :: in_filename, lat_filename

namelist / beam_track_params / lat_filename, beam_init, ran_seed, spin, n_warmup, &
                               space_charge, sc_ds, sc_mesh

! defaults
lat_filename = 'lat.bmad'
beam_init%n_particle = 1
beam_init%n_bunch = 1
ran_seed = 12345
spin = .false.
n_warmup = 1
space_charge = .false.
sc_ds = 0
sc_mesh = [32, 32, 32]

n_arg = command_argument_count()
in_filename = 'bmad_driver.in'
if (n_arg >= 1) call get_command_argument(1, in_filename)

open (1, file = in_filename, status = 'old')
read (1, nml = beam_track_params)
close (1)

call ran_seed_put (ran_seed)
bmad_com%spin_tracking_on = spin

call bmad_parser (lat_filename, lat)
ele1 => lat%ele(0)
ele2 => lat%ele(lat%n_ele_track)
gamma = ele1%value(e_tot$) / mass_of(lat%param%particle)

! warm-up tracks (NOT timed) -- amortize first-call/allocation (and the SC FFT plan / PTC init)
do i = 1, max(n_warmup, 1)
  call init_beam_distribution (ele1, lat%param, beam_init, beam)
  if (space_charge) call apply_sc_kick (beam, sc_ds, sc_mesh, gamma)
  call track_beam (lat, beam, ele1, ele2, err)
end do

! timed track on a fresh, identical beam. For space charge: ONE IGF solve (asymmetric kick) over the
! full drift, then the drift transport -- matches ImpactX's kick->drift, one solve.
call init_beam_distribution (ele1, lat%param, beam_init, beam)
call system_clock (c1, crate)
if (space_charge) call apply_sc_kick (beam, sc_ds, sc_mesh, gamma)
call track_beam (lat, beam, ele1, ele2, err)
call system_clock (c2)
dt_ns = real(c2 - c1, rp) / real(crate, rp) * 1.0e9_rp

call report (beam, dt_ns, spin)

contains

!------------------------------------------------------------------------------
! One 3D space-charge kick via OpenSpaceCharge -- integrated Green function (Ryne PRSTAB),
! open-boundary Hockney FFT with CIC deposit/gather -- the SAME model as ImpactX/HELIX, applied
! exactly ONCE over ds. This is byte-for-byte the physics of Bmad's native apply_fft_3d_kicks
! (csr_and_space_charge_mod), executed a single time so the benchmark compares 1 solve to 1 solve.
subroutine apply_sc_kick (beam, ds, mesh, gamma)
type (beam_struct), target :: beam
real(rp) ds, gamma
integer mesh(3)
type (mesh3d_struct) :: mesh3d
type (coord_struct), pointer :: p(:)
real(rp), allocatable :: xs(:), ys(:), zs(:), qs(:), Eb(:,:)
integer j, na, n
real(rp) factor, pz0, ef, dpz, new_beta

p => beam%bunch(1)%particle
n = size(p)
allocate (xs(n), ys(n), zs(n), qs(n))
na = 0
do j = 1, n
  if (p(j)%state /= alive$) cycle
  na = na + 1
  xs(na) = p(j)%vec(1); ys(na) = p(j)%vec(3); zs(na) = p(j)%vec(5); qs(na) = p(j)%charge
end do
if (na == 0) return

mesh3d%nhi = mesh
mesh3d%gamma = gamma
call deposit_particles (xs(1:na), ys(1:na), zs(1:na), mesh3d, qa = qs(1:na))
call space_charge_3d (mesh3d)
allocate (Eb(3, na))
call interpolate_field_batch (xs(1:na), ys(1:na), zs(1:na), mesh3d, na, E = Eb)

na = 0
do j = 1, n
  if (p(j)%state /= alive$) cycle
  na = na + 1
  factor = ds / (p(j)%p0c * p(j)%beta)
  pz0 = sqrt((1 + p(j)%vec(6))**2 - p(j)%vec(2)**2 - p(j)%vec(4)**2)
  p(j)%vec(2) = p(j)%vec(2) + Eb(1,na) * factor / mesh3d%gamma**2
  p(j)%vec(4) = p(j)%vec(4) + Eb(2,na) * factor / mesh3d%gamma**2
  ef = Eb(3,na) * factor
  dpz = sqrt_alpha(1 + p(j)%vec(6), ef*ef + 2*ef*pz0)
  p(j)%vec(6) = p(j)%vec(6) + dpz
  call convert_pc_to (p(j)%p0c * (1 + p(j)%vec(6)), p(j)%species, beta = new_beta)
  p(j)%vec(5) = p(j)%vec(5) * new_beta / p(j)%beta
  p(j)%beta = new_beta
end do

end subroutine

!------------------------------------------------------------------------------
subroutine report (beam, dt_ns, spin)
type (beam_struct), target :: beam
real(rp) dt_ns
logical spin
type (coord_struct), pointer :: p(:)
integer j, n
real(rp) x, px, y, py, z
real(rp) sx, s2x, spx, s2px, sxpx, sy, s2y, spy, s2py, sypy, sz, s2z
real(rp) ssx, ssy, ssz, s2sx, s2sy, s2sz
real(rp) sig_x, sig_y, sig_t, emit_x, emit_y, sig_sx, sig_sy, sig_sz

p => beam%bunch(1)%particle
n = 0
sx=0; s2x=0; spx=0; s2px=0; sxpx=0
sy=0; s2y=0; spy=0; s2py=0; sypy=0
sz=0; s2z=0
ssx=0; ssy=0; ssz=0; s2sx=0; s2sy=0; s2sz=0

do j = 1, size(p)
  if (p(j)%state /= alive$) cycle
  n = n + 1
  x = p(j)%vec(1); px = p(j)%vec(2); y = p(j)%vec(3); py = p(j)%vec(4); z = p(j)%vec(5)
  sx=sx+x; s2x=s2x+x*x; spx=spx+px; s2px=s2px+px*px; sxpx=sxpx+x*px
  sy=sy+y; s2y=s2y+y*y; spy=spy+py; s2py=s2py+py*py; sypy=sypy+y*py
  sz=sz+z; s2z=s2z+z*z
  if (spin) then
    ssx=ssx+p(j)%spin(1); s2sx=s2sx+p(j)%spin(1)**2
    ssy=ssy+p(j)%spin(2); s2sy=s2sy+p(j)%spin(2)**2
    ssz=ssz+p(j)%spin(3); s2sz=s2sz+p(j)%spin(3)**2
  endif
end do

if (n == 0) n = 1
write (*, '(a, i0, a)') 'Track: ', nint(dt_ns, 8), 'ns'

if (spin) then
  sig_sx = sqrt(max(s2sx/n - (ssx/n)**2, 0.0_rp))
  sig_sy = sqrt(max(s2sy/n - (ssy/n)**2, 0.0_rp))
  sig_sz = sqrt(max(s2sz/n - (ssz/n)**2, 0.0_rp))
  write (*, '(a, es16.9, a, es16.9, a, es16.9, a)') &
        'Validate: {"sigma_sx": ', sig_sx, ', "sigma_sy": ', sig_sy, &
        ', "sigma_sz": ', sig_sz, '}'
else
  sig_x = sqrt(max(s2x/n - (sx/n)**2, 0.0_rp))
  sig_y = sqrt(max(s2y/n - (sy/n)**2, 0.0_rp))
  sig_t = sqrt(max(s2z/n - (sz/n)**2, 0.0_rp))
  emit_x = sqrt(max((s2x/n - (sx/n)**2)*(s2px/n - (spx/n)**2) - (sxpx/n - (sx/n)*(spx/n))**2, 0.0_rp))
  emit_y = sqrt(max((s2y/n - (sy/n)**2)*(s2py/n - (spy/n)**2) - (sypy/n - (sy/n)*(spy/n))**2, 0.0_rp))
  write (*, '(a, es16.9, a, es16.9, a, es16.9, a, es16.9, a, es16.9, a)') &
        'Validate: {"sigma_x": ', sig_x, ', "sigma_y": ', sig_y, &
        ', "sigma_t": ', sig_t, ', "emit_x": ', emit_x, ', "emit_y": ', emit_y, '}'
endif

end subroutine

end program

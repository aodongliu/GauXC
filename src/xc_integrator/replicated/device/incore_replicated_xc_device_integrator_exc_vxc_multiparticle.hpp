/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#pragma once
#include "incore_replicated_xc_device_integrator.hpp"
#include "device/local_device_work_driver.hpp"
#include "device/xc_device_aos_data.hpp"
#include <gauxc/exceptions.hpp>
#include <gauxc/util/unused.hpp>

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef GAUXC_MP_DEVICE_TRACE
#include <iostream>
#endif

// WP2a-5 attribution instrumentation -- off by default, compiles to nothing
// unless the driver is built with GAUXC_ENABLE_NVTX_MP=ON (CMakeLists.txt).
// One NVTX range per (species|pair, phase) lane of the species-serial loop
// below, named "MP:s<p>:<phase>" / "MP:pair<i>:C" / "MP:s<p>:D:<S|Z>".
#ifdef GAUXC_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#endif

namespace GauXC  {
namespace detail {

/******************************************************************************
 *   Device (CUDA) MultiParticle (NEO) EXC/VXC -- design Phase-2 §1.4-§1.7    *
 *                                                                            *
 *  This is the species-serial generalization of                              *
 *  `incore_replicated_xc_device_integrator_exc_vxc.hpp`, and it is a          *
 *  line-for-line device transcription of the host oracle                     *
 *  `reference_replicated_xc_host_integrator_exc_vxc_multiparticle.hpp`.      *
 *                                                                            *
 *  Structure of one task batch (phase ordering B -> C -> D is NOT optional:   *
 *  the intra XC kernel OVERWRITES vrho while the EPC stage ACCUMULATES into   *
 *  it):                                                                      *
 *                                                                            *
 *    A  per-species density   (host oracle :257-364)                          *
 *    B  per-species intra XC  (host oracle :372-423)   [overwrites vrho]      *
 *    C  inter-species EPC     (host oracle :426-482)   [accumulates vrho]     *
 *    D  per-species Z + VXC   (host oracle :485-554)                          *
 *                                                                            *
 *  Every stage but C reuses an existing single-species LWD entry verbatim;    *
 *  the species it applies to is a property of the *data object* (design axiom *
 *  2), selected through `species_scope` (axiom 1/RAII).                       *
 ******************************************************************************/

namespace mp_device_detail {

/** Alpha-only eligibility (design §1.5), decided on the host, once per call,
 *  purely from the data -- never from a caller flag.
 *
 *  ChronusQ forces distinguishable quantum protons high-spin, so Pbeta = 0 and
 *  Ps == Pz *bitwise*.  When that holds and the species carries no intra
 *  functional, rho_minus is identically 0.0 and both VXC channels are bitwise
 *  equal, so one density matrix, one accumulator and one assembly lane suffice.
 *
 *  Two conditions beyond §1.5's literal predicate:
 *   - LD == NBF, because the device path requires contiguous nbf x nbf storage
 *     anyway and the memcmp would otherwise compare padding.
 *   - the species must not be the *electron* of an active pair.  §1.5's
 *     exactness argument (step 3: "EPC adds particle derivative to the +
 *     channel only") is a statement about the particle side; on the electron
 *     side the NEO convention scatters into BOTH spin channels, which makes
 *     rho_minus's potential non-zero and VXCs != VXCz.  Tightening the
 *     predicate keeps the silent-fallback-is-exact property intact.
 */
inline bool alpha_only_eligible( const double* Ps, int64_t ldps,
                                 const double* Pz, int64_t ldpz,
                                 bool has_intra, bool is_pair_electron,
                                 int64_t nbf ) {
  if( not Pz or has_intra or is_pair_electron ) return false;
  if( ldps != nbf or ldpz != nbf ) return false;
  return std::memcmp( Ps, Pz, size_t(nbf) * size_t(nbf) * sizeof(double) ) == 0;
}

#ifdef GAUXC_ENABLE_NVTX
/** RAII NVTX range for one (species|pair, phase) lane.  Syncs the master
 *  queue before popping so the range's reported wall time reflects actual
 *  device completion, not host dispatch latency -- the driver already runs
 *  every MP kernel on a single master stream (strict issue-order execution),
 *  so this sync changes nothing about what runs where, only when the host
 *  learns it finished.  Used only when profiling; never on the default
 *  (uninstrumented) hot path measured for G6. */
struct nvtx_phase_guard {
  DeviceBackend* be;
  nvtx_phase_guard( DeviceBackend* b, const char* name ) : be(b) {
    nvtxRangePushA(name);
  }
  ~nvtx_phase_guard() {
    if( be ) be->master_queue_synchronize();
    nvtxRangePop();
  }
};
#define GAUXC_MP_NVTX(be, ...)                                              \
  char _gauxc_nvtx_buf[40];                                                 \
  std::snprintf(_gauxc_nvtx_buf, sizeof(_gauxc_nvtx_buf), __VA_ARGS__);     \
  mp_device_detail::nvtx_phase_guard _gauxc_nvtx_g(be, _gauxc_nvtx_buf)
#else
#define GAUXC_MP_NVTX(be, ...) do {} while(0)
#endif

} // namespace mp_device_detail


template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  eval_exc_vxc_( const std::vector<multiparticle_density>& densities,
                 const MultiParticleFunctionalSpec& functional_spec,
                 const MultiParticleXCTerms& terms,
                 std::vector<multiparticle_vxc>& vxc,
                 value_type* intra_exc,
                 value_type* inter_pair_exc,
                 const IntegratorSettingsXC& ks_settings ) {

  const size_t np     = densities.size();
  const size_t ninter = functional_spec.inter_functionals.size();

  // ---- Validation.  Transcribed from the host entry point (:34-82), plus the
  //      device-only requirement that every leading dimension equal NBF (the
  //      device stores dense nbf x nbf matrices and copies them wholesale).
  if( np == 0 )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC/VXC requires at least one density");
  if( np != vxc.size() )
    GAUXC_GENERIC_EXCEPTION("MultiParticle density/VXC size mismatch");
  if( this->load_balancer_->basis_count() != np )
    GAUXC_GENERIC_EXCEPTION("MultiParticle density count must match LoadBalancer basis count");
  if( functional_spec.intra_functionals.size() > np )
    GAUXC_GENERIC_EXCEPTION("Too many MultiParticle intra functional entries");

  for( auto p : terms.active_intra )
    if( p >= functional_spec.intra_functionals.size() )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle active intra index");
  for( auto i : terms.active_inter )
    if( i >= ninter )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle active inter index");
  for( auto p : terms.vxc_targets )
    if( p >= np )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle VXC target index");

  std::vector<bool> build_vxc(np, false);
  for( auto p : terms.vxc_targets ) build_vxc[p] = true;

  for( size_t p = 0; p < np; ++p ) {
    const int64_t nbf = this->load_balancer_->basis(p).nbf();
    const auto& den   = densities[p];
    const bool  has_z = den.Pz != nullptr;
    if( den.m != den.n )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Ps must be square");
    if( den.m != nbf )
      GAUXC_GENERIC_EXCEPTION("MultiParticle Ps dimension must match its basis");
    if( den.ldps < nbf )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDPS");
    if( den.Pz and den.ldpz < nbf )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDPZ");
    if( build_vxc[p] and (not vxc[p].VXCs or vxc[p].ldvxcs < nbf) )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDVXCS");
    if( build_vxc[p] and has_z and (not vxc[p].VXCz or vxc[p].ldvxcz < nbf) )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle LDVXCZ");

    // Device-only: dense nbf x nbf storage on both sides of every H2D / D2H
    if( den.ldps != nbf )
      GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC requires LDPS == NBF");
    if( den.Pz and den.ldpz != nbf )
      GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC requires LDPZ == NBF");
    if( build_vxc[p] and vxc[p].ldvxcs != nbf )
      GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC requires LDVXCS == NBF");
    if( build_vxc[p] and has_z and vxc[p].ldvxcz != nbf )
      GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC requires LDVXCZ == NBF");
  }

  for( size_t i = 0; i < ninter; ++i ) {
    const auto& pair = functional_spec.inter_functionals[i];
    if( pair.electron >= np or pair.particle >= np )
      GAUXC_GENERIC_EXCEPTION("Invalid MultiParticle inter functional index");
  }

  // ---- Reductions are host-side only in 2a, mirroring the host oracle
  //      (:96-97).  A device-memory reducer (NCCL) would have to reduce a
  //      per-species accumulator set and is deliberately out of scope.
  if( not this->reduction_driver_->takes_host_memory() )
    GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC Only Works With Host Reductions");
  if( this->reduction_driver_->takes_device_memory() )
    GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC does not support device-memory reductions");

  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();

  // Allocate Device memory
  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get());
  if( not lwd )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC/VXC requires a device local work driver");
  if( not lwd->supports_multiparticle() )
    GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC is not implemented for this local work driver");

  auto rt = detail::as_device_runtime(this->load_balancer_->runtime());
  auto device_data_ptr = lwd->create_device_data(rt);

  GAUXC_MPI_CODE( MPI_Barrier(rt.comm()); )

  // Compute local contributions to EXC / VXC and retrieve them from the device
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    multiparticle_exc_vxc_local_work_( densities, functional_spec, terms, vxc,
      intra_exc, inter_pair_exc, ks_settings, tasks.begin(), tasks.end(),
      *device_data_ptr );
  });

  GAUXC_MPI_CODE(
  this->timer_.time_op("XCIntegrator.ImbalanceWait_EXC_VXC",[&](){
    MPI_Barrier(this->load_balancer_->runtime().comm());
  });
  )

  // Reduce results in host memory (host oracle :99-109)
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){
    for( size_t p = 0; p < np; ++p ) {
      if( not build_vxc[p] ) continue;
      const int64_t nbf = this->load_balancer_->basis(p).nbf();
      this->reduction_driver_->allreduce_inplace( vxc[p].VXCs, nbf * nbf, ReductionOp::Sum );
      if( densities[p].Pz )
        this->reduction_driver_->allreduce_inplace( vxc[p].VXCz, nbf * nbf, ReductionOp::Sum );
    }

    this->reduction_driver_->allreduce_inplace( intra_exc, np, ReductionOp::Sum );
    if( ninter )
      this->reduction_driver_->allreduce_inplace( inter_pair_exc, ninter, ReductionOp::Sum );
  });

}


template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  multiparticle_exc_vxc_local_work_(
    const std::vector<multiparticle_density>& densities,
    const MultiParticleFunctionalSpec& functional_spec,
    const MultiParticleXCTerms& terms,
    std::vector<multiparticle_vxc>& vxc,
    value_type* intra_exc,
    value_type* inter_pair_exc,
    const IntegratorSettingsXC& settings,
    host_task_iterator task_begin, host_task_iterator task_end,
    XCDeviceData& device_data ) {

  // The KS settings carry nothing the multiparticle device path consumes; the
  // host oracle likewise resolves and ignores them (:129-133).
  GauXC::util::unused(settings);

  const size_t np     = densities.size();
  const size_t ninter = functional_spec.inter_functionals.size();

  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get());
  if( not lwd )
    GAUXC_GENERIC_EXCEPTION("MultiParticle EXC/VXC requires a device local work driver");

  std::vector<bool> active_intra(np, false);
  std::vector<bool> active_inter(ninter, false);
  std::vector<bool> build_vxc(np, false);
  for( auto p : terms.active_intra ) active_intra[p] = true;
  for( auto i : terms.active_inter ) active_inter[i] = true;
  for( auto p : terms.vxc_targets )  build_vxc[p]    = true;

  //--------------------------------------------------------------------------
  //  Resolve the immutable multiparticle descriptor ONCE (design axiom 5).
  //  Mirrors the host oracle's derivation loop (:142-190).
  //--------------------------------------------------------------------------
  multiparticle_tracker mp;
  mp.species.resize(np);
  mp.pairs.resize(ninter);
  mp.do_vxc = not terms.vxc_targets.empty();

  // Which species carry an intra functional, and of what kind
  std::vector<const functional_type*> intra_func(np, nullptr);

  for( size_t p = 0; p < np; ++p ) {
    const auto& basis = this->load_balancer_->basis(p);
    auto& s = mp.species[p];
    s.index     = p;
    s.nbf       = static_cast<int32_t>(basis.nbf());
    s.nshells   = static_cast<int32_t>(basis.nshells());
    s.scheme    = densities[p].Pz ? UKS : RKS;
    s.xmat_fac  = (s.scheme == RKS) ? 2.0 : 1.0;
    s.approx    = LDA;
    s.build_vxc = build_vxc[p];

    const auto* funcs = p < functional_spec.intra_functionals.size() ?
      &functional_spec.intra_functionals[p] : nullptr;

    if( active_intra[p] and funcs and not funcs->empty() ) {
      // Design §1.4 declared restriction: a single intra functional per
      // species.  Supporting several needs per-species eps/vrho/vgamma
      // accumulation scratch that nothing in the shipped ChronusQ path or in
      // any Phase-1 reference exercises.
      if( funcs->size() > 1 )
        GAUXC_GENERIC_EXCEPTION("Device MultiParticle: multiple intra functionals per species NYI");

      const auto& func = *funcs->front();
      if( func.is_mgga() )
        GAUXC_GENERIC_EXCEPTION("MultiParticle mGGA intra-XC is not implemented");

      s.has_intra    = true;
      s.participates = true;
      s.approx       = func.is_gga() ? GGA : LDA;
      intra_func[p]  = funcs->front().get();
    }
  }

  std::vector<bool> is_pair_electron(np, false);
  for( size_t i = 0; i < ninter; ++i ) {
    const auto& pair = functional_spec.inter_functionals[i];
    auto& pd = mp.pairs[i];
    pd.electron = pair.electron;
    pd.particle = pair.particle;
    pd.active   = active_inter[i] and not pair.functionals.empty();

    for( const auto& func : pair.functionals ) {
      if( func->is_gga() or func->is_mgga() )
        GAUXC_GENERIC_EXCEPTION("MultiParticle GGA/mGGA inter-XC is not implemented");
    }
    // Same restriction, same reason, as for the intra functionals: §1.6's
    // pack/eval/scatter runs one ExchCXX evaluation per pair.
    if( pd.active and pair.functionals.size() > 1 )
      GAUXC_GENERIC_EXCEPTION("Device MultiParticle: multiple inter functionals per pair NYI");

    if( pd.active ) {
      mp.species[pair.electron].participates = true;
      mp.species[pair.particle].participates = true;
      is_pair_electron[pair.electron] = true;
    }
    inter_pair_exc[i] = 0.0;
  }

  // Alpha-only proton channel (§1.5): a data-derived specialization with a
  // silent, exact two-channel fallback.
  for( size_t p = 0; p < np; ++p ) {
    auto& s = mp.species[p];
    if( s.scheme != UKS ) continue;
    s.alpha_only = mp_device_detail::alpha_only_eligible(
      densities[p].Ps, densities[p].ldps, densities[p].Pz, densities[p].ldpz,
      s.has_intra, is_pair_electron[p], static_cast<int64_t>(s.nbf) );
  }

  for( size_t p = 0; p < np; ++p ) intra_exc[p] = 0.0;

#ifdef GAUXC_MP_DEVICE_TRACE
  for( size_t p = 0; p < np; ++p ) {
    const auto& s = mp.species[p];
    std::cout << "[GAUXC MP TRACE] species " << p
              << " nbf=" << s.nbf
              << " scheme=" << (s.scheme == UKS ? "UKS" : "RKS")
              << " approx=" << (s.approx == GGA ? "GGA" : "LDA")
              << " xmat_fac=" << s.xmat_fac
              << " intra=" << (s.has_intra ? 1 : 0)
              << " vxc=" << (s.build_vxc ? 1 : 0)
              << " part=" << (s.participates ? 1 : 0)
              << " route=" << (not s.participates ? "inactive" :
                               s.alpha_only       ? "ALPHA-ONLY" :
                               s.scheme == UKS    ? "two-channel" : "scalar")
              << std::endl;
  }
  for( size_t i = 0; i < ninter; ++i )
    std::cout << "[GAUXC MP TRACE] pair " << i << " (" << mp.pairs[i].electron
              << "," << mp.pairs[i].particle << ") active="
              << (mp.pairs[i].active ? 1 : 0) << std::endl;
#endif

  //--------------------------------------------------------------------------
  //  Task ordering (design R5): stable_sort on a permutation-invariant key.
  //  The key is a SUM over participating species of that species' quadratic
  //  work estimate, so relabelling the species cannot change it; `stable_sort`
  //  then makes the surviving order a deterministic function of the load
  //  balancer's own output rather than of libstdc++'s introsort pivots.
  //--------------------------------------------------------------------------
  {
    std::vector<size_t> part_species;
    for( size_t p = 0; p < np; ++p )
      if( mp.species[p].participates ) part_species.push_back(p);

    auto task_cost = [&]( const XCTask& task ) {
      size_t w = 0;
      for( auto p : part_species ) {
        const size_t nbe = task.basis_screening(p).nbe;
        w += nbe * (1 + nbe);
      }
      return w * task.points.size();
    };
    std::stable_sort( task_begin, task_end,
      [&]( const XCTask& a, const XCTask& b ) {
        return task_cost(a) > task_cost(b);
      });
  }

  // Check that Partition Weights have been calculated
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored )
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified");

  // Basis maps, one per species -- taken from the LoadBalancer, which already
  // owns one per basis.  Non-participating species contribute none.
  std::vector<const BasisSetMap*> basis_maps(np, nullptr);
  for( size_t p = 0; p < np; ++p )
    if( mp.species[p].participates )
      basis_maps[p] = &this->load_balancer_->basis_map(p);

  // Zero the caller's VXC blocks (host oracle :172-175).  A species that is a
  // VXC target but participates in nothing -- no intra functional, no active
  // pair -- is never touched on the device and must still come back as zero.
  if( mp.do_vxc )
  for( size_t p = 0; p < np; ++p ) {
    if( not mp.species[p].build_vxc ) continue;
    const size_t nbf = static_cast<size_t>(mp.species[p].nbf);
    std::fill( vxc[p].VXCs, vxc[p].VXCs + nbf * nbf, 0.0 );
    if( densities[p].Pz )
      std::fill( vxc[p].VXCz, vxc[p].VXCz + nbf * nbf, 0.0 );
  }

  //--------------------------------------------------------------------------
  //  Static device state
  //--------------------------------------------------------------------------
  device_data.init_species( np );           // implies reset_allocations()
  device_data.allocate_static_data_exc_vxc_multiparticle( mp );

  for( size_t p = 0; p < np; ++p ) {
    if( not mp.species[p].participates ) continue;
    device_density_channels ch;
    ch.Ps   = densities[p].Ps;
    ch.ldps = static_cast<int32_t>(densities[p].ldps);
    // The alpha-only channel stores ONE density matrix; its Z channel is Ps
    // bitwise, which is exactly what made it eligible.
    if( not mp.species[p].alpha_only ) {
      ch.Pz   = densities[p].Pz;
      ch.ldpz = static_cast<int32_t>(densities[p].ldpz);
    }
    device_data.send_static_data_density_basis_species( p, ch,
      this->load_balancer_->basis(p) );
  }

  device_data.zero_exc_vxc_integrands_multiparticle( mp );
  device_data.populate_submat_maps_multiparticle( mp, task_begin, task_end,
    basis_maps );

  // The per-batch vrho zeroing (design §1.4 step A / WP2A2 D11) needs the grid
  // arrays of the live species; `XCDeviceStackData` owns them and the LWD
  // reaches them the same way.
  auto* stack_data = dynamic_cast<XCDeviceStackData*>(&device_data);
  if( not stack_data )
    GAUXC_GENERIC_EXCEPTION("Device MultiParticle EXC/VXC requires a stack XCDeviceData");

  auto zero_grid_array = [&]( double* ptr, size_t n, const char* msg ) {
    if( ptr ) stack_data->device_backend_->set_zero_async_master_queue( n, ptr, msg );
  };

  //--------------------------------------------------------------------------
  //  The species-serial batch loop (design §1.4)
  //--------------------------------------------------------------------------
  auto task_it = task_begin;
  while( task_it != task_end ) {

    const auto batch_begin = task_it;
    task_it = device_data.generate_buffers_multiparticle( mp, basis_maps,
      task_it, task_end );

    const size_t npts_batch = stack_data->total_npts_task_batch;

    // How many tasks of THIS batch each species is active on.  This mirrors
    // the data layer's compaction predicate exactly (`nbe > 0`), and it is
    // the length of that species' `host_device_tasks`.  A species can be
    // active on none of a batch's tasks -- the multi-basis load balancer keeps
    // a task when *any* species screens in -- and then it has no device task
    // array at all, so not one of the per-species kernels below may be
    // launched for it.  Its per-point arrays must still read defined values,
    // because the EPC stage consumes den_s / den_z over the whole batch.
    std::vector<size_t> nactive_tasks(np, 0);
    for( auto it = batch_begin; it != task_it; ++it ) {
      const XCTask& t = *it;   // const overload: never materializes screenings
      for( size_t p = 0; p < np; ++p )
        if( mp.species[p].participates and t.basis_screening(p).nbe > 0 )
          nactive_tasks[p]++;
    }

    //---------------- A: per-species density (host oracle :257-364) ----------
    for( size_t p = 0; p < np; ++p ) {
      const auto& s = mp.species[p];
      if( not s.participates ) continue;
      species_scope scope( &device_data, p );
      GAUXC_MP_NVTX(stack_data->device_backend_, "MP:s%zu:A", p);

      const bool is_gga         = s.approx == GGA;
      const bool two_channel    = (s.scheme == UKS) and not s.alpha_only;

      if( nactive_tasks[p] == 0 ) {
        // `eval_vvars_*` would have zeroed these; nothing else can.
        auto& idle = stack_data->base_stack;
        zero_grid_array( idle.den_s_eval_device, npts_batch, "MP idle den_s Zero" );
        zero_grid_array( idle.den_z_eval_device, npts_batch, "MP idle den_z Zero" );
        continue;
      }

      if( is_gga ) lwd->eval_collocation_gradient( &device_data );
      else         lwd->eval_collocation( &device_data );

      auto do_xmat_vvar = [&]( density_id den_id ) {
        lwd->eval_xmat( s.xmat_fac, &device_data, false, den_id );
        if( is_gga ) lwd->eval_vvars_gga( &device_data, den_id );
        else         lwd->eval_vvars_lda( &device_data, den_id );
      };

      do_xmat_vvar( DEN_S );

      // The U variables turn (rho_s, rho_z) into (rho_alpha, rho_beta) and
      // build gamma.  The alpha-only channel skips the stage entirely: with
      // Ps == Pz bitwise, rho_alpha IS rho_s and rho_beta is exactly 0.0, so
      // `den_s` already holds the total density the EPC stage wants (§1.5).
      auto& live = stack_data->base_stack;
      if( two_channel ) {
        do_xmat_vvar( DEN_Z );
        // Every point at which this species is screened out must read a
        // defined gamma: `eval_vvars_*` zeroes the whole batch density array,
        // but `eval_uvars_gga` only writes the points its (compact) task list
        // covers.
        if( is_gga ) {
          zero_grid_array( live.gamma_pp_eval_device, npts_batch, "MP gamma++ Zero" );
          zero_grid_array( live.gamma_pm_eval_device, npts_batch, "MP gamma+- Zero" );
          zero_grid_array( live.gamma_mm_eval_device, npts_batch, "MP gamma-- Zero" );
          lwd->eval_uvars_gga( &device_data, UKS );
        } else {
          lwd->eval_uvars_lda( &device_data, UKS );
        }
      } else if( not s.alpha_only ) {
        if( is_gga ) {
          zero_grid_array( live.gamma_eval_device, npts_batch, "MP gamma Zero" );
          lwd->eval_uvars_gga( &device_data, RKS );
        } else {
          lwd->eval_uvars_lda( &device_data, RKS );   // LDA+RKS: a no-op kernel
        }
      }

      // A species with no intra functional has nothing to overwrite vrho, so
      // the EPC stage must accumulate into a zeroed array.
      if( not s.has_intra ) {
        if( two_channel or s.alpha_only ) {
          zero_grid_array( live.vrho_pos_eval_device, npts_batch, "MP vrho+ Zero" );
          zero_grid_array( live.vrho_neg_eval_device, npts_batch, "MP vrho- Zero" );
        } else {
          zero_grid_array( live.vrho_eval_device, npts_batch, "MP vrho Zero" );
        }
      }
    }

    //---------------- B: per-species intra XC (host oracle :372-423) ---------
    //                 OVERWRITES vrho -- must precede C.
    for( size_t p = 0; p < np; ++p ) {
      const auto& s = mp.species[p];
      if( not s.has_intra or nactive_tasks[p] == 0 ) continue;
      species_scope scope( &device_data, p );
      GAUXC_MP_NVTX(stack_data->device_backend_, "MP:s%zu:B", p);

      if( s.approx == GGA ) lwd->eval_kern_exc_vxc_gga( *intra_func[p], &device_data );
      else                  lwd->eval_kern_exc_vxc_lda( *intra_func[p], &device_data );

      lwd->inc_exc( &device_data );
      lwd->inc_nel( &device_data );
    }

    //---------------- C: inter-species EPC (host oracle :426-482) ------------
    //                 ACCUMULATES into vrho -- must follow B.
    for( size_t i = 0; i < ninter; ++i ) {
      if( not mp.pairs[i].active ) continue;
      GAUXC_MP_NVTX(stack_data->device_backend_, "MP:pair%zu:C", i);
      const auto& func = *functional_spec.inter_functionals[i].functionals.front();
      lwd->eval_kern_exc_vxc_inter_lda( func, &device_data, mp, i );
      lwd->inc_inter_exc( &device_data, mp, i );
    }

    //---------------- D: per-species Z matrix + VXC (host oracle :485-554) ---
    if( mp.do_vxc )
    for( size_t p = 0; p < np; ++p ) {
      const auto& s = mp.species[p];
      if( not s.participates or not s.build_vxc ) continue;
      if( nactive_tasks[p] == 0 ) continue;   // no device task array to launch on
      species_scope scope( &device_data, p );

      // The Z-matrix kernel is selected by the species' API-level KS scheme,
      // NOT by its storage shape: the alpha-only channel is stored RKS-shaped
      // but assembled with the stock UKS kernel, which reads vrho_pos and
      // vrho_neg (the latter identically 0.0) and therefore produces exactly
      // the Z matrix the two-channel path would.
      const auto zmat_scheme  = s.scheme;
      const bool two_channel  = (s.scheme == UKS) and not s.alpha_only;

      auto do_zmat_vxc = [&]( density_id den_id ) {
        GAUXC_MP_NVTX(stack_data->device_backend_, "MP:s%zu:D:%s", p,
          den_id == DEN_S ? "S" : "Z");
        if( s.approx == GGA ) lwd->eval_zmat_gga_vxc( &device_data, zmat_scheme, den_id );
        else                  lwd->eval_zmat_lda_vxc( &device_data, zmat_scheme, den_id );
        lwd->inc_vxc( &device_data, den_id, false );
      };

      do_zmat_vxc( DEN_S );
      if( two_channel ) do_zmat_vxc( DEN_Z );
    }

  } // Loop over batches of batches

  //--------------------------------------------------------------------------
  //  Symmetrize (host oracle :568-582) and retrieve (design §1.7)
  //--------------------------------------------------------------------------
  if( mp.do_vxc )
  for( size_t p = 0; p < np; ++p ) {
    const auto& s = mp.species[p];
    if( not s.participates or not s.build_vxc ) continue;
    species_scope scope( &device_data, p );
    lwd->symmetrize_vxc( &device_data, DEN_S );
    if( (s.scheme == UKS) and not s.alpha_only )
      lwd->symmetrize_vxc( &device_data, DEN_Z );
  }

  auto rt = detail::as_device_runtime(this->load_balancer_->runtime());
  rt.device_backend()->master_queue_synchronize();

  std::vector<double> N_EL(np, 0.0);   // per-species, diagnostic only (§1.7)
  std::vector<device_vxc_channels> vxc_channels(np);
  for( size_t p = 0; p < np; ++p ) {
    if( not (mp.do_vxc and mp.species[p].build_vxc) ) continue;
    vxc_channels[p].VXCs   = vxc[p].VXCs;
    vxc_channels[p].ldvxcs = static_cast<int32_t>(vxc[p].ldvxcs);
    // A UKS species -- two-channel or alpha-only -- returns both channels at
    // the API boundary.  For the alpha-only channel the data layer mirrors the
    // single accumulator into VXCz on the host, bitwise (§1.5).
    if( densities[p].Pz ) {
      vxc_channels[p].VXCz   = vxc[p].VXCz;
      vxc_channels[p].ldvxcz = static_cast<int32_t>(vxc[p].ldvxcz);
    }
  }

  this->timer_.time_op("XCIntegrator.DeviceToHostCopy_EXC_VXC",[&](){
    device_data.retrieve_exc_vxc_integrands_multiparticle( mp, intra_exc,
      inter_pair_exc, N_EL.data(), vxc_channels );
  });

}

}
}

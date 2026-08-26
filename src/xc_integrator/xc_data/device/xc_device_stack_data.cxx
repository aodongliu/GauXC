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
#include "xc_device_stack_data.hpp"
#include "buffer_adaptor.hpp"
#include <gauxc/runtime_environment.hpp>

namespace GauXC {

namespace detail {
  size_t memory_cap() {
    if( getenv("GAUXC_DEVICE_MEMORY_CAP" ) ) {
      return std::stoull( getenv("GAUXC_DEVICE_MEMORY_CAP") );
    } else { return std::numeric_limits<size_t>::max(); }
  }
}

XCDeviceStackData::XCDeviceStackData(const DeviceRuntimeEnvironment& rt) :
  runtime_(rt) { 
    device_ptr = runtime_.device_memory();
    devmem_sz  = runtime_.device_memory_size();
    device_backend_ = runtime_.device_backend();
    reset_allocations(); 
  }





XCDeviceStackData::~XCDeviceStackData() noexcept = default;


double* XCDeviceStackData::vxc_s_device_data() { return static_stack.vxc_s_device; }
double* XCDeviceStackData::vxc_z_device_data() { return static_stack.vxc_z_device; }
double* XCDeviceStackData::vxc_y_device_data() { return static_stack.vxc_y_device; }
double* XCDeviceStackData::vxc_x_device_data() { return static_stack.vxc_x_device; }
double* XCDeviceStackData::exc_device_data() { return static_stack.exc_device; }
double* XCDeviceStackData::nel_device_data() { return static_stack.nel_device; }
double* XCDeviceStackData::exx_k_device_data() { return static_stack.exx_k_device; }
double* XCDeviceStackData::fxc_s_device_data() { return static_stack.fxc_s_device; }
double* XCDeviceStackData::fxc_z_device_data() { return static_stack.fxc_z_device; }
double* XCDeviceStackData::fxc_y_device_data() { return static_stack.fxc_y_device; }
double* XCDeviceStackData::fxc_x_device_data() { return static_stack.fxc_x_device; }

device_queue XCDeviceStackData::queue() { 
  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  return device_backend_->queue();
}




void XCDeviceStackData::reset_allocations() {
  dynmem_ptr = device_ptr;
  dynmem_sz  = devmem_sz;
  allocated_terms.reset();
  static_stack.reset();
  base_stack.reset();
  mp_static.reset();
  mp_dyn.reset();
  mp_ninter_        = 0;
  mp_active_        = false;
  mp_grid_owner_    = true;
  active_alpha_only_ = false;
  mp_batch_static_req_ = 0;
  mp_batch_est_bytes_  = 0;
  mp_batch_dyn_used_   = 0;
  mp_batch_ntasks_     = 0;
}


// ---------------------------------------------------------------------------
//  Multi-species (NEO / multiparticle) context machinery -- design §1.3
// ---------------------------------------------------------------------------

void XCDeviceStackData::store_species_state( size_t p ) {
  auto& s = stack_species_.at(p);
  s.global_dims     = global_dims;
  s.allocated_terms = allocated_terms;
  s.static_stack    = static_stack;
  s.base_stack      = base_stack;
  s.alpha_only      = active_alpha_only_;
}

void XCDeviceStackData::load_species_state( size_t p ) {
  const auto& s = stack_species_.at(p);
  global_dims        = s.global_dims;
  allocated_terms    = s.allocated_terms;
  static_stack       = s.static_stack;
  base_stack         = s.base_stack;
  active_alpha_only_ = s.alpha_only;
}

void XCDeviceStackData::resize_species_slots( size_t np ) {
  stack_species_.clear();
  stack_species_.resize(np);
}

void XCDeviceStackData::init_species( size_t np ) {
  if( np == 0 ) GAUXC_GENERIC_EXCEPTION("At least one species is required");
  reset_allocations();        // virtual: resets every level's live state
  global_dims = allocated_dims{};
  resize_species_slots(np);   // virtual: resets every level's slots
  nspecies_          = np;
  active_species_    = 0;
  active_alpha_only_ = false;
}

void XCDeviceStackData::select_species( size_t p ) {
  if( p >= nspecies_ )
    GAUXC_GENERIC_EXCEPTION("Requested species index is out of range");
  if( p == active_species_ ) return;
  if( stack_species_.size() != nspecies_ )
    GAUXC_GENERIC_EXCEPTION("Species contexts have not been initialized");
  store_species_state( active_species_ );
  load_species_state( p );
  active_species_ = p;
}

const XCDeviceStackData::stack_species_state&
  XCDeviceStackData::species_state( size_t p ) {
  if( p >= nspecies_ )
    GAUXC_GENERIC_EXCEPTION("Requested species index is out of range");
  if( stack_species_.size() != nspecies_ )
    GAUXC_GENERIC_EXCEPTION("Species contexts have not been initialized");
  // The live members are authoritative for the active species -- mirror them
  // into the slot so callers always see current pointers.
  if( p == active_species_ ) store_species_state( p );
  return stack_species_.at(p);
}

const XCTask::screening_data&
  XCDeviceStackData::host_bfn_screening( const XCTask& t ) const {
  // With a single species context this IS the legacy expression, so every
  // single-species sizing/packing site is unchanged by construction.
  if( nspecies_ == 1 ) return t.bfn_screening;
  if( t.bfn_screenings.empty() ) {
    if( active_species_ == 0 ) return t.bfn_screening;
    GAUXC_GENERIC_EXCEPTION("Requested basis screening is not available");
  }
  return t.bfn_screenings.at(active_species_);
}

XCTask::screening_data&
  XCDeviceStackData::host_bfn_screening( XCTask& t ) const {
  if( nspecies_ == 1 ) return t.bfn_screening;
  if( t.bfn_screenings.empty() ) {
    if( active_species_ == 0 ) return t.bfn_screening;
    GAUXC_GENERIC_EXCEPTION("Requested basis screening is not available");
  }
  return t.bfn_screenings.at(active_species_);
}


void XCDeviceStackData::allocate_static_data_exc_vxc_multiparticle(
  const multiparticle_tracker& mp ) {

  const size_t np = mp.nspecies();
  if( np != nspecies_ )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: species count does not match init_species");

  // One arena, carved sequentially across species -- never per-species pools.
  for( size_t p = 0; p < np; ++p ) {
    const auto& s = mp.species.at(p);
    if( not s.participates ) continue;
    select_species(p);
    active_alpha_only_ = s.alpha_only;
    allocate_static_data_exc_vxc( s.nbf, s.nshells, mp.species_terms(p),
      s.build_vxc );
  }
  select_species(0);

  // Shared inter-pair storage (§1.7)
  mp_ninter_ = mp.pairs.size();
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );
  mp_static.acc_scr_device = mem.aligned_alloc<double>( 1, csl );
  if( mp_ninter_ )
    mp_static.inter_exc_device = mem.aligned_alloc<double>( mp_ninter_, csl );

  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft();
}


void XCDeviceStackData::send_static_data_density_basis_species( size_t p,
  const device_density_channels& den, const BasisSet<double>& basis ) {

  species_scope scope( this, p );

  if( not allocated_terms.exc_vxc )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: Density/Basis Not Stack Allocated");
  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  if( not den.Ps )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: missing scalar density channel");

  const auto nbf = global_dims.nbf;
  if( (size_t)basis.nbf() != nbf )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: basis does not match allocated NBF");

  if( den.ldps != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPs must be NBF");
  device_backend_->copy_async( nbf*nbf, den.Ps, static_stack.dmat_s_device,
    "MP P_scalar H2D" );

  // A two-channel species has a Z density matrix allocated; the alpha-only
  // channel (§1.5) deliberately does not, and its Pz is exactly Ps.
  if( static_stack.dmat_z_device ) {
    if( not den.Pz )
      GAUXC_GENERIC_EXCEPTION("MultiParticle: missing Z density channel");
    if( den.ldpz != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPz must be NBF");
    device_backend_->copy_async( nbf*nbf, den.Pz, static_stack.dmat_z_device,
      "MP P_z H2D" );
  }

  device_backend_->copy_async( basis.nshells(), basis.data(),
    static_stack.shells_device, "MP Shells H2D" );

  device_backend_->master_queue_synchronize();
}


void XCDeviceStackData::zero_exc_vxc_integrands_multiparticle(
  const multiparticle_tracker& mp ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  const size_t np = mp.nspecies();
  if( np != nspecies_ )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: species count does not match init_species");

  for( size_t p = 0; p < np; ++p ) {
    if( not mp.species.at(p).participates ) continue;
    species_scope scope( this, p );
    zero_exc_vxc_integrands( mp.species_terms(p) );
  }

  if( mp_ninter_ )
    device_backend_->set_zero( mp_ninter_, mp_static.inter_exc_device,
      "MP Inter EXC Zero" );
}


void XCDeviceStackData::retrieve_exc_vxc_integrands_multiparticle(
  const multiparticle_tracker& mp, double* intra_EXC, double* inter_EXC,
  double* N_EL, const std::vector<device_vxc_channels>& vxc ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  const size_t np = mp.nspecies();
  if( np != nspecies_ )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: species count does not match init_species");
  if( mp.do_vxc and vxc.size() != np )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: one VXC channel set per species is required");

  // Alpha-only species produce a single accumulator which is bitwise the Z
  // channel as well (§1.5/§1.7): the D2H lands in VXCs and is mirrored on the
  // host once every device copy has completed.
  struct alpha_mirror { const double* src; double* dst; size_t n; };
  std::vector<alpha_mirror> mirrors;

  for( size_t p = 0; p < np; ++p ) {
    const auto& s = mp.species.at(p);
    if( not s.participates ) continue;
    species_scope scope( this, p );

    const auto nbf = global_dims.nbf;

    if( intra_EXC )
      device_backend_->copy_async( 1, static_stack.exc_device, intra_EXC + p,
        "MP EXC D2H" );
    if( N_EL )
      device_backend_->copy_async( 1, static_stack.nel_device, N_EL + p,
        "MP NEL D2H" );

    if( not (mp.do_vxc and s.build_vxc) ) continue;
    const auto& ch = vxc.at(p);

    if( ch.VXCs ) {
      if( ch.ldvxcs != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDVXCs must be NBF");
      device_backend_->copy_async( nbf*nbf, static_stack.vxc_s_device, ch.VXCs,
        "MP VXCs D2H" );
    }

    if( ch.VXCz ) {
      if( ch.ldvxcz != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDVXCz must be NBF");
      if( static_stack.vxc_z_device ) {
        device_backend_->copy_async( nbf*nbf, static_stack.vxc_z_device,
          ch.VXCz, "MP VXCz D2H" );
      } else if( s.alpha_only ) {
        if( not ch.VXCs )
          GAUXC_GENERIC_EXCEPTION("MultiParticle: alpha-only VXCz requires VXCs");
        mirrors.push_back( alpha_mirror{ ch.VXCs, ch.VXCz, nbf*nbf } );
      } else {
        GAUXC_GENERIC_EXCEPTION("MultiParticle: no Z VXC accumulator allocated");
      }
    }
  }

  if( mp_ninter_ and inter_EXC )
    device_backend_->copy_async( mp_ninter_, mp_static.inter_exc_device,
      inter_EXC, "MP Inter EXC D2H" );

  // Unlike the single-species retrieval this synchronizes: the alpha-only
  // host mirror must observe the completed D2H of the scalar channel.
  device_backend_->master_queue_synchronize();
  for( const auto& m : mirrors ) std::copy_n( m.src, m.n, m.dst );
}


void XCDeviceStackData::populate_submat_maps_multiparticle(
  const multiparticle_tracker& mp,
  host_task_iterator task_begin, host_task_iterator task_end,
  const std::vector<const BasisSetMap*>& basis_maps ) {

  const size_t np = mp.nspecies();
  if( np != nspecies_ )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: species count does not match init_species");
  if( basis_maps.size() != np )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: one basis map per species is required");

  for( size_t p = 0; p < np; ++p ) {
    const auto& s = mp.species.at(p);
    if( not s.participates ) continue;
    if( not basis_maps[p] )
      GAUXC_GENERIC_EXCEPTION("MultiParticle: missing basis map for a participating species");
    species_scope scope( this, p );
    populate_submat_maps( s.nbf, task_begin, task_end, *basis_maps[p] );
  }
}


XCDeviceStackData::host_task_iterator
  XCDeviceStackData::generate_buffers_multiparticle(
    const multiparticle_tracker& mp,
    const std::vector<const BasisSetMap*>& basis_maps,
    host_task_iterator task_begin, host_task_iterator task_end ) {

  const size_t np = mp.nspecies();
  if( np != nspecies_ )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: species count does not match init_species");
  if( basis_maps.size() != np )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: one basis map per species is required");

  // Which species owns the shared grid arrays (points + weights)?  Every
  // species evaluates the SAME immutable molecular grid, so exactly one copy
  // is allocated and packed and the rest alias it.
  size_t grid_owner = np;
  for( size_t p = 0; p < np; ++p )
    if( mp.species.at(p).participates ) { grid_owner = p; break; }
  if( grid_owner == np )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: no participating species");

  // Cache per-species term trackers once (axiom 5: no re-resolution in a loop)
  std::vector<integrator_term_tracker> terms(np);
  for( size_t p = 0; p < np; ++p )
    if( mp.species.at(p).participates ) terms[p] = mp.species_terms(p);

  const size_t species_state_reentry = active_species_;
  mp_active_ = true;

  // ---- 1. Static (batch-size independent) reservation, summed over species
  size_t static_req = 0;
  for( size_t p = 0; p < np; ++p ) {
    if( not mp.species.at(p).participates ) continue;
    select_species(p);
    static_req += get_static_mem_requirement();
  }
  select_species(grid_owner);

  if( static_req > dynmem_sz ) {
    mp_active_ = false;
    GAUXC_GENERIC_EXCEPTION("Insufficient memory to even start!");
  }
  size_t mem_left = dynmem_sz - static_req;

  // ---- 2. Batch size: EVERY participating species must be resident for the
  //         same task, so the per-task requirement is the sum over species.
  //         Failing to make this species-aware is design risk R1.
  const size_t mp_dyn_bytes_per_point =
    mp_dyn_doubles_per_point * sizeof(double);
  size_t est_bytes = 0;
  host_task_iterator task_it = task_begin;
  while( task_it != task_end ) {

    size_t mem_req_batch = 0;
    for( size_t p = 0; p < np; ++p ) {
      if( not mp.species.at(p).participates ) continue;
      select_species(p);
      mp_grid_owner_ = ( p == grid_owner );
      mem_req_batch += get_mem_req( terms[p], *task_it );
    }
    // Shared inter-species scratch (§1.6) + slop for its aligned allocations
    mem_req_batch += mp_dyn_bytes_per_point * task_it->points.size()
                   + mp_dyn_nalloc * 256;

    if( mem_req_batch > mem_left ) break;
    mem_left  -= mem_req_batch;
    est_bytes += mem_req_batch;
    task_it++;
  }
  select_species(grid_owner);
  mp_grid_owner_ = true;

  if( task_it == task_begin ) {
    mp_active_ = false;
    GAUXC_GENERIC_EXCEPTION("Insufficient device memory for a single multiparticle task");
  }

  // ---- 3. Carve the arena once, sequentially, across all species
  device_buffer_t buf{ dynmem_ptr, dynmem_sz };
  double* shared_points_x = nullptr;
  double* shared_points_y = nullptr;
  double* shared_points_z = nullptr;
  double* shared_weights  = nullptr;

  for( size_t p = 0; p < np; ++p ) {
    if( not mp.species.at(p).participates ) continue;
    if( not basis_maps[p] )
      GAUXC_GENERIC_EXCEPTION("MultiParticle: missing basis map for a participating species");

    select_species(p);
    mp_grid_owner_ = ( p == grid_owner );

    buf = allocate_dynamic_stack( terms[p], task_begin, task_it, buf );

    if( mp_grid_owner_ ) {
      shared_points_x = base_stack.points_x_device;
      shared_points_y = base_stack.points_y_device;
      shared_points_z = base_stack.points_z_device;
      shared_weights  = base_stack.weights_device;
    } else {
      // Alias the single packed copy of the immutable grid into this species'
      // task descriptors instead of allocating and copying identical arrays.
      base_stack.points_x_device = shared_points_x;
      base_stack.points_y_device = shared_points_y;
      base_stack.points_z_device = shared_points_z;
      base_stack.weights_device  = shared_weights;
    }

    pack_and_send( terms[p], task_begin, task_it, *basis_maps[p] );
  }
  select_species(grid_owner);
  mp_grid_owner_ = true;

  // ---- 4. Shared inter-species scratch from whatever remains
  mp_dyn.reset();
  {
    auto [ptr, sz] = buf;
    buffer_adaptor mem( ptr, sz );
    const size_t msz = total_npts_task_batch;
    const size_t aln = 256;
    mp_dyn.rho_lhs_device       = mem.aligned_alloc<double>(   msz, aln, csl );
    mp_dyn.rho_rhs_device       = mem.aligned_alloc<double>(   msz, aln, csl );
    mp_dyn.pair_den_device      = mem.aligned_alloc<double>( 2*msz, aln, csl );
    mp_dyn.pair_eps_device      = mem.aligned_alloc<double>(   msz, aln, csl );
    mp_dyn.pair_vrho_device     = mem.aligned_alloc<double>( 2*msz, aln, csl );
    mp_dyn.pair_vrho_lhs_device = mem.aligned_alloc<double>(   msz, aln, csl );
    mp_dyn.pair_vrho_rhs_device = mem.aligned_alloc<double>(   msz, aln, csl );
    mp_dyn.wtf_device           = mem.aligned_alloc<double>(   msz, aln, csl );
    buf = device_buffer_t{ mem.stack(), mem.nleft() };
  }

  {
    auto [ptr, sz] = buf;
    (void)sz;
    mp_batch_dyn_used_ = (size_t)( (char*)ptr - (char*)dynmem_ptr );
  }
  mp_batch_static_req_ = static_req;
  mp_batch_est_bytes_  = est_bytes;
  mp_batch_ntasks_     = std::distance( task_begin, task_it );

  mp_active_ = false;
  select_species(species_state_reentry);
  return task_it;
}

void XCDeviceStackData::allocate_static_data_weights( int32_t natoms ) {

  if( allocated_terms.weights ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack Weights");

  // Save state
  global_dims.natoms  = natoms;

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.coords_device = mem.aligned_alloc<double>( 3 * natoms, csl );

  // Allow for RAB to be strided and properly aligned
  const auto ldatoms   = get_ldatoms();
  const auto rab_align = get_rab_align();
  static_stack.rab_device = mem.aligned_alloc<double>( natoms * ldatoms, rab_align, csl );

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 

  allocated_terms.weights = true;
}

void XCDeviceStackData::allocate_static_data_exc_vxc( int32_t nbf, int32_t nshells, integrator_term_tracker enabled_terms, bool do_vxc ) {

  if( allocated_terms.exc_vxc ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack EXC VXC");
  if( enabled_terms.ks_scheme == _UNDEF_SCHEME )
    GAUXC_GENERIC_EXCEPTION("Must have a KS Scheme set to allocate Stack EXC VXC");

  // Save state
  global_dims.nshells = nshells;
  global_dims.nbf     = nbf; 

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device     = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.exc_device        = mem.aligned_alloc<double>( 1 , csl);
  static_stack.nel_device        = mem.aligned_alloc<double>( 1 , csl);
  static_stack.acc_scr_device    = mem.aligned_alloc<double>( 1 , csl);
  
  allocated_terms.ks_scheme = enabled_terms.ks_scheme;
  static_stack.dmat_s_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
  if( not (allocated_terms.ks_scheme == RKS) ) {
      static_stack.dmat_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      if( allocated_terms.ks_scheme == GKS ) {
        static_stack.dmat_y_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.dmat_x_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      }
  }

  if( do_vxc ) {
    static_stack.vxc_s_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
    if( not (allocated_terms.ks_scheme == RKS) ) {
        static_stack.vxc_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        if( allocated_terms.ks_scheme == GKS ) {
          static_stack.vxc_y_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
          static_stack.vxc_x_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        }
    }
  }

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 
    

  allocated_terms.exc_vxc = true;
}

void XCDeviceStackData::allocate_static_data_fxc_contraction( int32_t nbf, int32_t nshells, integrator_term_tracker enabled_terms ) {

  if( allocated_terms.fxc_contraction ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack FXC Contraction");
  if( enabled_terms.ks_scheme == _UNDEF_SCHEME )
    GAUXC_GENERIC_EXCEPTION("Must have a KS Scheme set to allocate Stack EXC VXC");

  // Save state
  global_dims.nshells = nshells;
  global_dims.nbf     = nbf; 

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device     = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.nel_device        = mem.aligned_alloc<double>( 1 , csl);
  static_stack.acc_scr_device    = mem.aligned_alloc<double>( 1 , csl);
  static_stack.dmat_s_device   = mem.aligned_alloc<double>( nbf * nbf , csl );
  static_stack.tdmat_s_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
  static_stack.fxc_s_device    = mem.aligned_alloc<double>( nbf * nbf , csl );
  
  allocated_terms.ks_scheme = enabled_terms.ks_scheme;
  if( not (allocated_terms.ks_scheme == RKS) ) {
      static_stack.dmat_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      static_stack.tdmat_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      static_stack.fxc_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      if( allocated_terms.ks_scheme == GKS ) {
        static_stack.dmat_y_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.dmat_x_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.tdmat_y_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.tdmat_x_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.fxc_y_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.fxc_x_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      }
  }

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 
    

  allocated_terms.fxc_contraction = true;
}

void XCDeviceStackData::allocate_static_data_den( int32_t nbf, int32_t nshells ) {

  if( allocated_terms.den ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack Density");

  // Save state
  global_dims.nshells = nshells;
  global_dims.nbf     = nbf; 

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device     = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.acc_scr_device    = mem.aligned_alloc<double>( 1 , csl);
  static_stack.nel_device        = mem.aligned_alloc<double>( 1 , csl);

  static_stack.dmat_s_device = mem.aligned_alloc<double>( nbf * nbf , csl);

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 

  allocated_terms.den = true;
}

void XCDeviceStackData::allocate_static_data_exc_grad( int32_t nbf, int32_t nshells, int32_t natoms, integrator_term_tracker enabled_terms ) {

  if( allocated_terms.exc_grad ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack EXC GRAD");

  // Save state
  global_dims.nshells = nshells;
  global_dims.nbf     = nbf; 
  global_dims.natoms  = natoms; 

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device     = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.exc_grad_device   = mem.aligned_alloc<double>( 3*natoms , csl);
  static_stack.nel_device        = mem.aligned_alloc<double>( 1 , csl);
  static_stack.acc_scr_device    = mem.aligned_alloc<double>( 1 , csl);

  allocated_terms.ks_scheme = enabled_terms.ks_scheme;
  static_stack.dmat_s_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
  if( not (allocated_terms.ks_scheme == RKS) ) {
      static_stack.dmat_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      if( allocated_terms.ks_scheme == GKS ) {
        static_stack.dmat_y_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.dmat_x_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      }
  }

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 

  allocated_terms.exc_grad = true;
}


void XCDeviceStackData::allocate_static_data_exx( int32_t nbf, int32_t nshells, size_t nshell_pairs, size_t nprim_pair_total, int32_t max_l ) {

  if( allocated_terms.exx ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack EXX");

  // Save state
  global_dims.nshells      = nshells;
  global_dims.nshell_pairs = nshell_pairs;
  global_dims.nprim_pairs  = nprim_pair_total;
  global_dims.nbf          = nbf; 
  global_dims.max_l        = max_l; 

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.prim_pairs_device = 
      mem.aligned_alloc<PrimitivePair<double>>(nprim_pair_total, csl);

  static_stack.exx_k_device = mem.aligned_alloc<double>( nbf * nbf , csl);
  static_stack.dmat_s_device  = mem.aligned_alloc<double>( nbf * nbf , csl);

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 

  allocated_terms.exx = true;
}

void XCDeviceStackData::allocate_static_data_exx_ek_screening( size_t ntasks, int32_t nbf, int32_t nshells, int nshell_pairs, int32_t max_l ) {

  if( allocated_terms.exx_ek_screening ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack EXX-EK Screening");

  // Save state
  global_dims.nshells      = nshells;
  global_dims.nshell_pairs = nshell_pairs;
  global_dims.nbf          = nbf; 
  global_dims.max_l        = max_l; 
  global_dims.ntask_ek     = ntasks;



  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.dmat_s_device   = mem.aligned_alloc<double>( nbf * nbf , csl);
  static_stack.ek_max_bfn_sum_device =
    mem.aligned_alloc<double>( ntasks , csl);
  static_stack.vshell_max_sparse_device = 
    mem.aligned_alloc<double>( nshell_pairs , csl);
  static_stack.shpair_row_ind_device = 
    mem.aligned_alloc<size_t>( nshell_pairs , csl);
  static_stack.shpair_col_ind_device = 
    mem.aligned_alloc<size_t>( nshell_pairs , csl);
  static_stack.ek_bfn_max_device = 
    mem.aligned_alloc<double>( nbf * ntasks , csl);
  static_stack.shell_to_bf_device =
    mem.aligned_alloc<int32_t>( nshells, csl );
  static_stack.shell_sizes_device =
    mem.aligned_alloc<int32_t>( nshells, csl );

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 

  allocated_terms.exx_ek_screening = true;
}






void XCDeviceStackData::send_static_data_weights( const Molecule& mol, const MolMeta& meta ) {

  if( not allocated_terms.weights ) 
    GAUXC_GENERIC_EXCEPTION("Weights Not Stack Allocated");

  const auto natoms = global_dims.natoms;
  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  // Copy Atomic Coordinates
  std::vector<double> coords( 3*natoms );
  for( auto i = 0ul; i < natoms; ++i ) {
    coords[ 3*i + 0 ] = mol[i].x;
    coords[ 3*i + 1 ] = mol[i].y;
    coords[ 3*i + 2 ] = mol[i].z;
  }
  device_backend_->copy_async( 3*natoms, coords.data(), static_stack.coords_device, 
    "Coords H2D" );

  // Invert and send RAB
  const auto ldatoms = get_ldatoms();
  std::vector<double> rab_inv(natoms*natoms);
  for( auto i = 0ul; i < (natoms*natoms); ++i) rab_inv[i] = 1./meta.rab().data()[i];
  device_backend_->copy_async_2d( natoms, natoms, rab_inv.data(), natoms,
    static_stack.rab_device, ldatoms, "RAB H2D" );

  device_backend_->master_queue_synchronize(); 
}

void XCDeviceStackData::send_static_data_density_basis( const double* Ps, int32_t ldps, const double* Pz, int32_t ldpz, const double* Py, int32_t ldpy, const double* Px, int32_t ldpx,
  const BasisSet<double>& basis ) {
  const bool is_gks = (Pz != nullptr) and (Py != nullptr) and (Px != nullptr);
  const bool is_uks = (Pz != nullptr) and (Py == nullptr) and (Px == nullptr);
  const bool is_rks = (Ps != nullptr) and (not is_uks and not is_gks);
  if( not is_rks and not is_uks and not is_gks )
    GAUXC_GENERIC_EXCEPTION("Densities do not match RKS, UKS, or GKS schemes");

  if( not (allocated_terms.exx or allocated_terms.exc_vxc or allocated_terms.exc_grad or allocated_terms.den or allocated_terms.exx_ek_screening or allocated_terms.fxc_contraction ) ) 
    GAUXC_GENERIC_EXCEPTION("Density/Basis Not Stack Allocated");

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");


  const auto nbf    = global_dims.nbf;
  // Check dimensions and copy density
  if( ldps != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPs must bf NBF");
  device_backend_->copy_async( nbf*nbf, Ps, static_stack.dmat_s_device, "P_scalar H2D" );
  if( not is_rks ) {
    if( ldpz != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPz must bf NBF");
    device_backend_->copy_async( nbf*nbf, Pz, static_stack.dmat_z_device, "P_z H2D" );
    if( is_gks ) {
      if( ldpy != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPy must bf NBF");
      if( ldpx != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPx must bf NBF");
      device_backend_->copy_async( nbf*nbf, Py, static_stack.dmat_y_device, "P_y H2D" );
      device_backend_->copy_async( nbf*nbf, Px, static_stack.dmat_x_device, "P_x H2D" );
    }
  }

  // Copy Basis Set
  device_backend_->copy_async( basis.nshells(), basis.data(), static_stack.shells_device,
    "Shells H2D" );

  device_backend_->master_queue_synchronize(); 
}


void XCDeviceStackData::send_static_data_trial_density(
  const double* tPs, int32_t ldtps, const double* tPz, int32_t ldtpz,
  const double* tPy, int32_t ldtpy, const double* tPx, int32_t ldtpx ) {

  const bool is_gks = (tPz != nullptr) && (tPy != nullptr) && (tPx != nullptr);
  const bool is_uks = (tPz != nullptr) && (tPy == nullptr) && (tPx == nullptr);
  const bool is_rks = (tPs != nullptr) && (not is_uks and not is_gks);
  if( not is_rks and not is_uks and not is_gks )
    GAUXC_GENERIC_EXCEPTION("Trial densities do not match RKS, UKS, or GKS schemes");

  if( not allocated_terms.fxc_contraction )
    GAUXC_GENERIC_EXCEPTION("Trial Density Not Stack Allocated");

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  const auto nbf = global_dims.nbf;
  // Check dimensions and copy density
  if( ldtps != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDTps must bf NBF");
  device_backend_->copy_async( nbf*nbf, tPs, static_stack.tdmat_s_device, "tP_scalar H2D" );
  if( not is_rks ) {
    if( ldtpz != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDTpz must bf NBF");
    device_backend_->copy_async( nbf*nbf, tPz, static_stack.tdmat_z_device, "tP_z H2D" );
    if( is_gks ) {
      if( ldtpy != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDTpy must bf NBF");
      if( ldtpx != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDTpx must bf NBF");
      device_backend_->copy_async( nbf*nbf, tPy, static_stack.tdmat_y_device, "tP_y H2D" );
      device_backend_->copy_async( nbf*nbf, tPx, static_stack.tdmat_x_device, "tP_x H2D" );
    }
  }
  
  device_backend_->master_queue_synchronize();
}


void XCDeviceStackData::send_static_data_shell_pairs( 
  const BasisSet<double>& basis,
  const ShellPairCollection<double>& shell_pairs ) {

  if( not allocated_terms.exx ) 
    GAUXC_GENERIC_EXCEPTION("ShellPairs Not Stack Allocated");

  const auto nshells = global_dims.nshells;
  if( shell_pairs.nshells() != nshells )
    GAUXC_GENERIC_EXCEPTION("Incompatible Basis for Stack Allocation");

  const auto nshell_pairs = global_dims.nshell_pairs;
  if( shell_pairs.npairs() != nshell_pairs )
    GAUXC_GENERIC_EXCEPTION("Incompatible ShellPairs for Stack Allocation");

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  // Copy primitive pairs
  std::vector<GauXC::PrimitivePair<double>> pp_host;
  for(const auto& sp : shell_pairs) {
    pp_host.insert( pp_host.end(), sp.prim_pairs(), sp.prim_pairs() + sp.nprim_pairs());
  }
  device_backend_->copy_async( global_dims.nprim_pairs, pp_host.data(),
    static_stack.prim_pairs_device, "PrimPairs H2D" );

  // Create SoA
  shell_pair_soa.reset();
  using point = XCDeviceShellPairSoA::point;
  const auto sp_row_ptr = shell_pairs.row_ptr();
  const auto sp_col_ind = shell_pairs.col_ind();

  shell_pair_soa.sp_row_ptr = sp_row_ptr;
  shell_pair_soa.sp_col_ind = sp_col_ind;

  GauXC::PrimitivePair<double>* prim_pair_ptr = static_stack.prim_pairs_device;
  for( auto i = 0ul, idx = 0ul; i < nshells; ++i ) {
    const auto j_st = sp_row_ptr[i];
    const auto j_en = sp_row_ptr[i+1];
    for( auto _j = j_st; _j < j_en; ++_j, idx++ ) {
      const auto j = sp_col_ind[_j];

      const auto& sp = shell_pairs.shell_pairs()[idx];
      const auto nprim_pairs = sp.nprim_pairs();
      shell_pair_soa.prim_pair_dev_ptr.emplace_back( prim_pair_ptr );
      prim_pair_ptr += nprim_pairs;

      shell_pair_soa.shell_pair_nprim_pairs.push_back(nprim_pairs);
      auto& bra = basis[i];
      auto& ket = basis[j];
      shell_pair_soa.shell_pair_shidx.emplace_back(i,j);
      shell_pair_soa.shell_pair_ls.emplace_back( bra.l(), ket.l());
      shell_pair_soa.shell_pair_centers.emplace_back(
        point{ bra.O()[0], bra.O()[1], bra.O()[2] },
        point{ ket.O()[0], ket.O()[1], ket.O()[2] }
      );
    }
  }
  
  device_backend_->master_queue_synchronize(); 
}

void XCDeviceStackData::send_static_data_exx_ek_screening( const double* V_max, 
  int32_t ldv, const BasisSetMap& basis_map, 
  const ShellPairCollection<double>& shpairs ) {

  if( not allocated_terms.exx_ek_screening ) 
    GAUXC_GENERIC_EXCEPTION("VMAX Not Stack Allocated");

  const auto nshells      = global_dims.nshells;
  const auto nshell_pairs = global_dims.nshell_pairs;
  if( ldv != (int)nshells ) GAUXC_GENERIC_EXCEPTION("LDV must bf NSHELLS");
  if( shpairs.npairs() != nshell_pairs ) 
    GAUXC_GENERIC_EXCEPTION("Inconsistent ShellPairs"); 
  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");


  // Pack VMAX
  std::vector<double> V_pack(nshell_pairs);
  const auto sp_row_ptr = shpairs.row_ptr();
  const auto sp_col_ind = shpairs.col_ind();
  for( auto i = 0ul; i < nshells; ++i ) {
    const auto j_st = sp_row_ptr[i];
    const auto j_en = sp_row_ptr[i+1];
    for( auto _j = j_st; _j < j_en; ++_j ) {
      const auto j = sp_col_ind[_j];
      V_pack[_j] = V_max[i + j*ldv];
    }
  }

  // Copy VMAX
  device_backend_->copy_async( nshell_pairs, V_pack.data(), 
    static_stack.vshell_max_sparse_device, "VMAX Sparse H2D");

  // Create sparse triplet for device
  std::vector<size_t> rowind(nshell_pairs);
  for( auto i = 0ul; i < nshells; ++i ) {
    const auto j_st = sp_row_ptr[i];
    const auto j_en = sp_row_ptr[i+1];
    for( auto _j = j_st; _j < j_en; ++_j ) {
      rowind[_j] = i;
    }
  }

  

  // Send adjacency
  device_backend_->copy_async( nshell_pairs, rowind.data(),
    static_stack.shpair_row_ind_device, "SP RowInd H2D");
  device_backend_->copy_async( nshell_pairs, sp_col_ind.data(),
    static_stack.shpair_col_ind_device, "SP ColInd H2D");

  std::vector<int32_t> shell2bf(nshells);
  std::vector<int32_t> shell_sizes(nshells);
  for(auto i = 0ul; i < nshells; ++i) {
    shell2bf[i] = basis_map.shell_to_first_ao(i);
    shell_sizes[i] = basis_map.shell_size(i);
  }
  

  device_backend_->copy_async( nshells, shell2bf.data(), static_stack.shell_to_bf_device,
    "Shell2BF H2D");
  device_backend_->copy_async( nshells, shell_sizes.data(), static_stack.shell_sizes_device,
    "ShellSizes H2D");
  
  device_backend_->master_queue_synchronize(); 

}


void XCDeviceStackData::zero_den_integrands() {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  device_backend_->set_zero( 1, static_stack.nel_device, "NEL Zero" );

}


void XCDeviceStackData::zero_exc_vxc_integrands(integrator_term_tracker enabled_terms) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  const auto nbf = global_dims.nbf;
  if(static_stack.vxc_s_device) device_backend_->set_zero( nbf*nbf, static_stack.vxc_s_device, "VXCs Zero" );
  if(static_stack.vxc_z_device) device_backend_->set_zero( nbf*nbf, static_stack.vxc_z_device, "VXCz Zero" );
  if(static_stack.vxc_y_device) device_backend_->set_zero( nbf*nbf, static_stack.vxc_y_device, "VXCy Zero" );
  if(static_stack.vxc_x_device) device_backend_->set_zero( nbf*nbf, static_stack.vxc_x_device, "VXCx Zero" );
  device_backend_->set_zero( 1,       static_stack.exc_device, "EXC Zero" );
  device_backend_->set_zero( 1,       static_stack.nel_device, "NEL Zero" );

}

void XCDeviceStackData::zero_fxc_contraction_integrands() {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  const auto nbf = global_dims.nbf;
  if(static_stack.fxc_s_device) device_backend_->set_zero( nbf*nbf, static_stack.fxc_s_device, "FXCs Zero" );
  if(static_stack.fxc_z_device) device_backend_->set_zero( nbf*nbf, static_stack.fxc_z_device, "FXCz Zero" );
  if(static_stack.fxc_y_device) device_backend_->set_zero( nbf*nbf, static_stack.fxc_y_device, "FXCy Zero" );
  if(static_stack.fxc_x_device) device_backend_->set_zero( nbf*nbf, static_stack.fxc_x_device, "FXCx Zero" );
  device_backend_->set_zero( 1,       static_stack.nel_device, "NEL Zero" );

}

void XCDeviceStackData::zero_exc_grad_integrands() {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  const auto natoms = global_dims.natoms;
  device_backend_->set_zero( 3*natoms, static_stack.exc_grad_device, "EXC Gradient Zero" );
  device_backend_->set_zero( 1,        static_stack.nel_device, "NEL Zero" );

}


void XCDeviceStackData::zero_exx_integrands() {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  const auto nbf = global_dims.nbf;
  device_backend_->set_zero( nbf*nbf, static_stack.exx_k_device, "K Zero" );

}

void XCDeviceStackData::zero_exx_ek_screening_intermediates() {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  const auto ntask_ek = global_dims.ntask_ek;
  const auto nbf      = global_dims.nbf;
  device_backend_->set_zero( ntask_ek*nbf, static_stack.ek_bfn_max_device, "EK BFNMAX Zero" );

}


void XCDeviceStackData::retrieve_exc_vxc_integrands( double* EXC, double* N_EL,
  double* VXCs, int32_t ldvxcs, double* VXCz, int32_t ldvxcz,
  double* VXCy, int32_t ldvxcy, double* VXCx, int32_t ldvxcx ) {

  const auto nbf = global_dims.nbf;

  device_backend_->copy_async( 1,       static_stack.nel_device, N_EL, "NEL D2H" );
  device_backend_->copy_async( 1,       static_stack.exc_device, EXC,  "EXC D2H" );

  if( ldvxcs and (ldvxcs != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDVXCs must be NBF");
  if( VXCs )
    device_backend_->copy_async( nbf*nbf, static_stack.vxc_s_device, VXCs,  "VXCs D2H" );

  if( ldvxcz and (ldvxcz != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDVXCz must be NBF");
  if( VXCz )
    device_backend_->copy_async( nbf*nbf, static_stack.vxc_z_device, VXCz,  "VXCz D2H" );

  if( ldvxcy and (ldvxcy != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDVXCy must be NBF");
  if( VXCy )
    device_backend_->copy_async( nbf*nbf, static_stack.vxc_y_device, VXCy,  "VXCy D2H" );

  if( ldvxcx and (ldvxcx != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDVXCx must be NBF");
  if( VXCx )
    device_backend_->copy_async( nbf*nbf, static_stack.vxc_x_device, VXCx,  "VXCx D2H" );

}

void XCDeviceStackData::retrieve_fxc_contraction_integrands( double* N_EL,
  double* FXCs, int32_t ldfxcs, double* FXCz, int32_t ldfxcz,
  double* FXCy, int32_t ldfxcy, double* FXCx, int32_t ldfxcx ) {

  const auto nbf = global_dims.nbf;
  device_backend_->copy_async( 1,       static_stack.nel_device, N_EL, "NEL D2H" );

  if( ldfxcs and (ldfxcs != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDFXCs must be NBF");
  if( FXCs )
    device_backend_->copy_async( nbf*nbf, static_stack.fxc_s_device, FXCs,  "FXCs D2H" );

  if( ldfxcz and (ldfxcz != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDFXCz must be NBF");
  if( FXCz )
    device_backend_->copy_async( nbf*nbf, static_stack.fxc_z_device, FXCz,  "FXCz D2H" );

  if( ldfxcy and (ldfxcy != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDFXCy must be NBF");
  if( FXCy )
    device_backend_->copy_async( nbf*nbf, static_stack.fxc_y_device, FXCy,  "FXCy D2H" );

  if( ldfxcx and (ldfxcx != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDFXCx must be NBF");
  if( FXCx )
    device_backend_->copy_async( nbf*nbf, static_stack.fxc_x_device, FXCx,  "FXCx D2H" );

}

void XCDeviceStackData::retrieve_den_integrands( double* N_EL ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  
  device_backend_->copy_async( 1, static_stack.nel_device, N_EL, "NEL D2H" );

}

void XCDeviceStackData::retrieve_exc_grad_integrands( double* EXC_GRAD, double* N_EL ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  
  const auto natoms = global_dims.natoms;
  device_backend_->copy_async( 3*natoms, static_stack.exc_grad_device, EXC_GRAD,  "EXC Gradient D2H" );
  device_backend_->copy_async( 1,        static_stack.nel_device,      N_EL,      "NEL D2H" );

}

void XCDeviceStackData::retrieve_exx_integrands( double* K, int32_t ldk ) {

  const auto nbf = global_dims.nbf;
  if( ldk != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDK must bf NBF");
  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  
  device_backend_->copy_async( nbf*nbf, static_stack.exx_k_device, K,  "K D2H" );

}

void XCDeviceStackData::retrieve_exx_ek_max_bfn_sum( double* MBS, int32_t nt ) {

  const auto ntask_ek = global_dims.ntask_ek;
  if( nt != (int)ntask_ek ) GAUXC_GENERIC_EXCEPTION("Inconsistent Task Count");
  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  device_backend_->copy_async( ntask_ek , static_stack.ek_max_bfn_sum_device, MBS, 
    "MBS D2H");

}






XCDeviceStackData::host_task_iterator XCDeviceStackData::generate_buffers(
  integrator_term_tracker terms,
  const BasisSetMap& basis_map,
  host_task_iterator task_begin,
  host_task_iterator task_end
) {

  if( get_static_mem_requirement() > dynmem_sz )
    GAUXC_GENERIC_EXCEPTION("Insufficient memory to even start!");

  size_t mem_left = dynmem_sz - get_static_mem_requirement();

  // Determine the number of batches that will fit into device memory
  host_task_iterator task_it = task_begin;
  while( task_it != task_end ) {

    // Get memory requirement for batch
    size_t mem_req_batch = get_mem_req( terms, *task_it );

    // Break out of loop if we can't allocate for this batch
    if( mem_req_batch > mem_left ) break;

    // Update remaining memory and increment task iterator
    mem_left -= mem_req_batch;
    task_it++;

  }

  // TODO: print this if verbose
  //std::cout << "XCDeviceStackData will allocate for " << std::distance(task_begin, task_it) << " Tasks MEMLEFT = " << mem_left << std::endl;

  // Pack host data and send to device
  allocate_dynamic_stack( terms, task_begin, task_it,
    device_buffer_t{dynmem_ptr, dynmem_sz} );

  pack_and_send( terms, task_begin, task_it, basis_map );

  return task_it;
}





size_t XCDeviceStackData::get_mem_req( 
  integrator_term_tracker terms,
  const host_task_type& task
) {

  const auto& points = task.points;
  const size_t npts  = points.size();

  required_term_storage reqt(terms);

  // Multiparticle: the grid points/weights are shared by every species and are
  // therefore charged to (and allocated by) exactly one of them.  Outside a
  // multiparticle batch `mp_grid_owner_` is always true.
  if( not mp_grid_owner_ ) {
    reqt.grid_points  = false;
    reqt.grid_weights = false;
  }

  size_t mem_req = 
    // Grid
    reqt.grid_points_size (npts)  * sizeof(double) + 
    reqt.grid_weights_size(npts)  * sizeof(double) +

    // U Variables
    reqt.grid_den_size(npts)      * sizeof(double) + 
    reqt.grid_den_grad_size(npts) * sizeof(double) +
    reqt.grid_lapl_size(npts)     * sizeof(double) +

    // H/K Matrices (GKS)
    reqt.grid_HK_size(npts)       * sizeof(double) +

    // V Variables
    reqt.grid_gamma_size(npts)    * sizeof(double) +
    reqt.grid_tau_size(npts)      * sizeof(double) +

    // XC output
    reqt.grid_eps_size(npts)      * sizeof(double) +
    reqt.grid_vrho_size(npts)     * sizeof(double) +
    reqt.grid_vgamma_size(npts)   * sizeof(double) +
    reqt.grid_vtau_size(npts)     * sizeof(double) +
    reqt.grid_vlapl_size(npts)    * sizeof(double) ;

    // second derivatives
    mem_req += 
    // U variables
    reqt.grid_tden_size(npts)      * sizeof(double) +
    reqt.grid_tden_grad_size(npts) * sizeof(double) +
    reqt.grid_tlapl_size(npts)     * sizeof(double) +
    reqt.grid_ttau_size(npts)      * sizeof(double) +
    // XC output
    reqt.grid_v2rho2_size(npts)    * sizeof(double) +
    reqt.grid_v2rhogamma_size(npts)  * sizeof(double) +
    reqt.grid_v2rholapl_size(npts)   * sizeof(double) +
    reqt.grid_v2rhotau_size(npts)  * sizeof(double) +
    reqt.grid_v2gamma2_size(npts) * sizeof(double) +
    reqt.grid_v2gammalapl_size(npts) * sizeof(double) +
    reqt.grid_v2gammatau_size(npts) * sizeof(double) +
    reqt.grid_v2lapl2_size(npts) * sizeof(double) +
    reqt.grid_v2lapltau_size(npts) * sizeof(double) +
    reqt.grid_v2tau2_size(npts) * sizeof(double) +
    // intermediate output
    reqt.grid_FXC_A_size(npts) * sizeof(double) +
    reqt.grid_FXC_B_size(npts) * sizeof(double) +
    reqt.grid_FXC_C_size(npts) * sizeof(double);

  // Alpha-only proton channel (§1.5): RKS-shaped storage plus the polarized
  // vrho_pos / vrho_neg arrays that the stock UKS Z-matrix kernel consumes.
  if( reqt.grid_vrho and alpha_only_vrho() )
    mem_req += 2 * npts * sizeof(double);

  return mem_req;
}









XCDeviceStackData::device_buffer_t XCDeviceStackData::allocate_dynamic_stack( 
  integrator_term_tracker terms,
  host_task_iterator task_begin, host_task_iterator task_end, 
  device_buffer_t buf ) {


  // Get total npts
  total_npts_task_batch = std::accumulate( task_begin, task_end, 0ul,
    [](const auto& a, const auto& b){ return a + b.points.size(); } );

  // Allocate device memory
  auto [ ptr, sz ] = buf;
  buffer_adaptor mem( ptr, sz );


  required_term_storage reqt(terms);

  // Multiparticle: the grid points/weights are allocated once by the owning
  // species; every other species aliases them (see
  // generate_buffers_multiparticle).  Always true outside a MP batch.
  if( not mp_grid_owner_ ) {
    reqt.grid_points  = false;
    reqt.grid_weights = false;
  }

  const size_t msz = total_npts_task_batch;
  const size_t aln = 256;
  
  const bool is_rks = terms.ks_scheme == RKS;
  const bool is_uks = terms.ks_scheme == UKS;
  const bool is_gks = terms.ks_scheme == GKS;
  const bool is_pol = is_uks or is_gks;
  const bool is_gga = terms.xc_approx == GGA;

  const bool is_den = terms.den;
  
  // Grid Points
  if( reqt.grid_points ) {
    base_stack.points_x_device = mem.aligned_alloc<double>( msz, aln, csl);
    base_stack.points_y_device = mem.aligned_alloc<double>( msz, aln, csl);
    base_stack.points_z_device = mem.aligned_alloc<double>( msz, aln, csl);
  }


  // Grid Weights
  if( reqt.grid_weights ) {
    base_stack.weights_device = mem.aligned_alloc<double>(msz, csl);
  }

  // Grid function evaluations
  if( reqt.grid_den ) { // Density 
    base_stack.den_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);

    if(is_pol) {
      base_stack.den_interleaved_device = mem.aligned_alloc<double>(2*msz, aln, csl);
      base_stack.den_z_eval_device      = mem.aligned_alloc<double>(msz, aln, csl);
    }

    if(is_gks){   
      base_stack.den_y_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      base_stack.den_x_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    }
  }

  if( reqt.grid_den_grad ) { // Density gradient
    base_stack.dden_sx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    base_stack.dden_sy_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    base_stack.dden_sz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 

    if(is_pol) { 
      base_stack.dden_zx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.dden_zy_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      base_stack.dden_zz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    }
    if( is_gks ) { 
      base_stack.dden_yx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.dden_yy_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      base_stack.dden_yz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      base_stack.dden_xx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.dden_xy_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      base_stack.dden_xz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    }
  }

  if( reqt.grid_tau ) { // Tau 
    base_stack.tau_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    if(is_pol) {
      base_stack.tau_interleaved_device = mem.aligned_alloc<double>(2*msz, aln, csl);
      base_stack.tau_z_eval_device      = mem.aligned_alloc<double>(msz, aln, csl);
    } 
  }

  if( reqt.grid_lapl ) { // Density Laplacian
    base_stack.lapl_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    if(is_pol) {
      base_stack.lapl_interleaved_device = mem.aligned_alloc<double>(2*msz, aln, csl);
      base_stack.lapl_z_eval_device      = mem.aligned_alloc<double>(msz, aln, csl);
    } 
  }

  if( reqt.grid_gamma ) { // Gamma
    if( is_pol  ) {  
      base_stack.gamma_eval_device    = mem.aligned_alloc<double>(3 * msz, aln, csl);
      base_stack.gamma_pp_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.gamma_pm_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.gamma_mm_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    } else {           
      base_stack.gamma_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }

  if( reqt.grid_vrho ) { // Vrho
    if( is_pol  ) { 
      base_stack.vrho_eval_device     = mem.aligned_alloc<double>(2 * msz, aln, csl);
      base_stack.vrho_pos_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.vrho_neg_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    } else {          
      base_stack.vrho_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      // Alpha-only proton channel (§1.5): single-channel vrho, but the stock
      // UKS Z-matrix kernel is reused verbatim and reads vrho_pos/vrho_neg.
      if( alpha_only_vrho() ) {
        base_stack.vrho_pos_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.vrho_neg_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }
  }

  if( reqt.grid_vgamma ) { // Vgamma
    if( is_pol  ) {  
      base_stack.vgamma_eval_device    = mem.aligned_alloc<double>(3*msz, aln, csl);
      base_stack.vgamma_pp_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.vgamma_pm_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.vgamma_mm_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    } else {
      base_stack.vgamma_eval_device    = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }

  if( is_gks ) {       // H, K matrices
    base_stack.K_x_eval_device   = mem.aligned_alloc<double>(msz, aln, csl);
    base_stack.K_y_eval_device   = mem.aligned_alloc<double>(msz, aln, csl);
    base_stack.K_z_eval_device   = mem.aligned_alloc<double>(msz, aln, csl);
    if( is_gga ) {
      base_stack.H_x_eval_device   = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.H_y_eval_device   = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.H_z_eval_device   = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }

  if( reqt.grid_eps ) { // Energy density 
    base_stack.eps_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
  }

  if( reqt.grid_vtau ) { // Vtau
    if( is_pol  ) { 
      base_stack.vtau_eval_device     = mem.aligned_alloc<double>(2 * msz, aln, csl);
      base_stack.vtau_pos_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.vtau_neg_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    } else {          
      base_stack.vtau_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }

  if( reqt.grid_vlapl ) { // Vlapl
    if( is_pol  ) { 
      base_stack.vlapl_eval_device     = mem.aligned_alloc<double>(2 * msz, aln, csl);
      base_stack.vlapl_pos_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.vlapl_neg_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    } else {          
      base_stack.vlapl_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }

  if( terms.fxc_contraction ) {
    // Trial density evaluation
    if( reqt.grid_tden ) { 
      base_stack.tden_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      if(is_pol) {
        base_stack.tden_z_eval_device      = mem.aligned_alloc<double>(msz, aln, csl);
      }
      if(is_gks){
        base_stack.tden_y_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
        base_stack.tden_x_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      }
    }

    // Trial density gradient
    if( reqt.grid_tden_grad ) {
      base_stack.tdden_sx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.tdden_sy_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.tdden_sz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 

      if(is_pol) { 
        base_stack.tdden_zx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.tdden_zy_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
        base_stack.tdden_zz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      }
      if( is_gks ) { 
        base_stack.tdden_yx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.tdden_yy_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
        base_stack.tdden_yz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
        base_stack.tdden_xx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.tdden_xy_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
        base_stack.tdden_xz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      }
    }

    // Trial tau
    if( reqt.grid_ttau ) {
      base_stack.ttau_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      if(is_pol) {
        base_stack.ttau_z_eval_device      = mem.aligned_alloc<double>(msz, aln, csl);
      } 
    }

    // Trial laplacian
    if( reqt.grid_tlapl ) {
      base_stack.tlapl_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      if(is_pol) {
        base_stack.tlapl_z_eval_device      = mem.aligned_alloc<double>(msz, aln, csl);
      } 
    }

    // Second derivatives of XC functional
    if( reqt.grid_v2rho2 ) {
      if( is_pol  ) { 
        base_stack.v2rho2_eval_device = mem.aligned_alloc<double>(3 * msz, aln, csl);
        base_stack.v2rho2_a_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rho2_a_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rho2_b_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2rho2_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2rhogamma ) {
      if( is_pol  ) { 
        base_stack.v2rhogamma_eval_device = mem.aligned_alloc<double>(6 * msz, aln, csl);
        base_stack.v2rhogamma_a_aa_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhogamma_a_ab_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhogamma_a_bb_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhogamma_b_aa_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhogamma_b_ab_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhogamma_b_bb_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2rhogamma_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2rholapl ) {
      if( is_pol  ) { 
        base_stack.v2rholapl_eval_device = mem.aligned_alloc<double>(4 * msz, aln, csl);
        base_stack.v2rholapl_a_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rholapl_a_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rholapl_b_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rholapl_b_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2rholapl_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2rhotau ) {
      if( is_pol  ) { 
        base_stack.v2rhotau_eval_device = mem.aligned_alloc<double>(4 * msz, aln, csl);
        base_stack.v2rhotau_a_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhotau_a_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhotau_b_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhotau_b_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2rhotau_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2gamma2 ) {
      if( is_pol  ) { 
        base_stack.v2gamma2_eval_device = mem.aligned_alloc<double>(6 * msz, aln, csl);
        base_stack.v2gamma2_aa_aa_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gamma2_aa_ab_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gamma2_aa_bb_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gamma2_ab_ab_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gamma2_ab_bb_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gamma2_bb_bb_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2gamma2_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2gammalapl ) {
      if( is_pol  ) { 
        base_stack.v2gammalapl_eval_device = mem.aligned_alloc<double>(6 * msz, aln, csl);
        base_stack.v2gammalapl_aa_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammalapl_aa_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammalapl_ab_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammalapl_ab_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammalapl_bb_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammalapl_bb_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2gammalapl_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2gammatau ) {
      if( is_pol  ) { 
        base_stack.v2gammatau_eval_device = mem.aligned_alloc<double>(6 * msz, aln, csl);
        base_stack.v2gammatau_aa_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammatau_aa_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammatau_ab_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammatau_ab_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammatau_bb_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammatau_bb_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2gammatau_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2lapl2 ) {
      if( is_pol  ) { 
        base_stack.v2lapl2_eval_device = mem.aligned_alloc<double>(3 * msz, aln, csl);
        base_stack.v2lapl2_a_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2lapl2_a_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2lapl2_b_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2lapl2_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2lapltau ) {
      if( is_pol  ) { 
        base_stack.v2lapltau_eval_device = mem.aligned_alloc<double>(4 * msz, aln, csl);
        base_stack.v2lapltau_a_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2lapltau_a_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2lapltau_b_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2lapltau_b_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2lapltau_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2tau2 ) {
      if( is_pol  ) { 
        base_stack.v2tau2_eval_device = mem.aligned_alloc<double>(3 * msz, aln, csl);
        base_stack.v2tau2_a_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2tau2_a_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2tau2_b_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2tau2_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    // Intermediate matrices for contraction
    if( reqt.grid_FXC_A ) {
      base_stack.FXC_A_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      if( is_pol  ) 
        base_stack.FXC_A_z_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    }
    
    if( reqt.grid_FXC_B ) {
      base_stack.FXC_Bx_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.FXC_By_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.FXC_Bz_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      if( is_pol  ) { 
        base_stack.FXC_Bx_z_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.FXC_By_z_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.FXC_Bz_z_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_FXC_C ) {
      base_stack.FXC_C_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      if( is_pol  ) 
        base_stack.FXC_C_z_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }



  // Update dynmem data for derived impls
  return device_buffer_t{ mem.stack(), mem.nleft() };
}

void XCDeviceStackData::pack_and_send( integrator_term_tracker terms,
  host_task_iterator task_begin, host_task_iterator task_end, const BasisSetMap& ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  // Multiparticle: only the grid-owning species packs and sends the shared,
  // immutable grid; every other species slices the same device arrays in its
  // own task descriptors.  Always the owner outside a MP batch.
  if( not mp_grid_owner_ ) {
    device_backend_->master_queue_synchronize();
    return;
  }

  // Host data packing arrays
  std::vector<double> points_x_pack, points_y_pack, points_z_pack;
  std::vector< double > weights_pack;

  // Contatenation utility
  auto concat_iterable = []( auto& a, const auto& b ) {
    a.insert( a.end(), b.begin(), b.end() );
  };

  // Pack points / weights
  for( auto it = task_begin; it != task_end; ++it ) {

    const auto& points  = it->points;
    const auto& weights = it->weights;

    //concat_iterable( points_pack,  points  );
    std::vector<double> pts_x, pts_y, pts_z;
    for( auto pt : points ) {
      pts_x.emplace_back( pt[0] );
      pts_y.emplace_back( pt[1] );
      pts_z.emplace_back( pt[2] );
    }
    concat_iterable( points_x_pack, pts_x );
    concat_iterable( points_y_pack, pts_y );
    concat_iterable( points_z_pack, pts_z );

    concat_iterable( weights_pack, weights );
    
  } // Loop over tasks

  if( points_x_pack.size() != total_npts_task_batch )
    GAUXC_GENERIC_EXCEPTION("Inconsistent Points-X allocation");
  if( points_y_pack.size() != total_npts_task_batch )
    GAUXC_GENERIC_EXCEPTION("Inconsistent Points-Y allocation");
  if( points_z_pack.size() != total_npts_task_batch )
    GAUXC_GENERIC_EXCEPTION("Inconsistent Points-Z allocation");
  if( weights_pack.size() != total_npts_task_batch )
    GAUXC_GENERIC_EXCEPTION("Inconsistent weights allocation");



  // Send grid data
  device_backend_->copy_async( points_x_pack.size(), points_x_pack.data(),
              base_stack.points_x_device, "send points_x buffer" );
  device_backend_->copy_async( points_y_pack.size(), points_y_pack.data(),
              base_stack.points_y_device, "send points_y buffer" );
  device_backend_->copy_async( points_z_pack.size(), points_z_pack.data(),
              base_stack.points_z_device, "send points_z buffer" );
  device_backend_->copy_async( weights_pack.size(), weights_pack.data(),
              base_stack.weights_device, "send weights buffer" );


  // Synchronize on the copy stream to keep host vecs in scope
  device_backend_->master_queue_synchronize(); 

}


void XCDeviceStackData::copy_weights_to_tasks( host_task_iterator task_begin, host_task_iterator task_end ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  // Sanity check that npts is consistent
  size_t local_npts = std::accumulate( task_begin, task_end, 0ul, 
    []( const auto& a, const auto& b ) { return a + b.points.size(); } );

  if( local_npts != total_npts_task_batch )
    GAUXC_GENERIC_EXCEPTION("NPTS Mismatch");

  // Copy weights into contiguous host data
  std::vector<double> weights_host(local_npts);
  device_backend_->copy_async( local_npts, base_stack.weights_device, 
    weights_host.data(), "Weights D2H" );
  device_backend_->master_queue_synchronize(); 

  // Place into host memory 
  auto* weights_ptr = weights_host.data();
  for( auto it = task_begin; it != task_end; ++it ) {
    const auto npts = it->points.size();
    std::copy_n( weights_ptr, npts, it->weights.data() );
    weights_ptr += npts;
  }

}


}

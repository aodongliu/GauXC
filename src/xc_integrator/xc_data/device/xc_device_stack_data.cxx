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
#include "device/common/syr2k_size_classes.hpp"
#include <gauxc/runtime_environment.hpp>

#include <algorithm>

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
  grid_npts_ = 0;
  ntasks_intra_ = 0;
  task_grid_offset_.clear();
  task_work_ranges_.clear();
  task_grid_offset_device_ = nullptr;
  w_times_f_device_        = nullptr;
  multiparticle_active_ = false;
  multiparticle_static.reset();
  pair_stack.reset();
  dynmem_ptr = device_ptr;
  dynmem_sz  = devmem_sz;
  allocated_terms.reset();
  static_stack.reset();
  base_stack.reset();
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

  const size_t npts  = task.npts;

  required_term_storage reqt(terms);

  // The multiparticle batch's grid is allocated and packed once by the caller
  if( shared_grid_ ) {
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

  return mem_req;
}









XCDeviceStackData::device_buffer_t XCDeviceStackData::allocate_dynamic_stack( 
  integrator_term_tracker terms,
  host_task_iterator task_begin, host_task_iterator task_end, 
  device_buffer_t buf ) {


  // Get total npts
  total_npts_task_batch = std::accumulate( task_begin, task_end, 0ul,
    [](const auto& a, const auto& b){ return a + size_t(b.npts); } );

  // Allocate device memory
  auto [ ptr, sz ] = buf;
  buffer_adaptor mem( ptr, sz );


  required_term_storage reqt(terms);

  // The multiparticle batch's grid is allocated by the caller, once.
  if( shared_grid_ ) {
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

  // The multiparticle batch's grid is packed by the caller, once
  if( shared_grid_ ) {
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




/******************************************************************************
 *                        The multiparticle XCtask list                       *

 In multiparticle (NEO) framework, each grid tasks now carry a list of screened basis.

 New idea introduced: Species task, which is grid task for specific species. 
 For example: (0, e):  grid task 0, electron 
              (0, p1): grid task 0, proton 1
              (1, p2): grid task 1, proton 2

In this function, we do the following:

Step 1. Build species tasks 
        Here we expand the grid task, and build XCTasks for each species of a grid task (which we call species task here). 
        Note we use a oncatenated shell list for all species tasks.

Step 2. Compute device memory for species tasks (they share the same grid)

Step 3. Partition tasks into batches. Add to a batch until an entire grid task cannot fit. Species tasks on the same grid are not split

Step 4. Re-order species tasks within a batch to put species with intra functional first (currently just electrons), then the rest
       (for contiguous memory access during functional evaluation and VXC syr2k launch grid)

 ******************************************************************************/

void XCDeviceStackData::build_species_tasks( integrator_term_tracker terms,
  const multiparticle_tracker& tracker,
  const std::vector<species_basis_range>& species_basis_ranges,
  const BasisSetMap& basis_map,
  host_task_iterator task_begin, host_task_iterator task_end,
  multiparticle_layout& layout ) {

  const size_t nspecies = tracker.nspecies();
  if( species_basis_ranges.size() != nspecies )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: one basis range per species is required");

  // For different sizes of syr2k dispatch. See Step 4
  multiparticle_active_ = true;

  // A quantum particle belongs to at most one active pair
  {
    std::vector<char> in_active_pair( nspecies, 0 );
    for( const auto& pair : tracker.pairs ) {
      if( not pair.active ) continue;
      if( in_active_pair[pair.particle] )
        GAUXC_GENERIC_EXCEPTION("MultiParticle: a quantum particle in more than one active pair is NYI");
      in_active_pair[pair.particle] = 1;
    }
  }

  layout.species_tasks.clear();
  layout.grid_offsets.clear();
  layout.point_offsets.clear();
  layout.species_indices.clear();
  layout.batches.clear();

  // Form an order for species that participates 
  // the species carrying an intra functional first, then the rest
  std::vector<size_t> intra_species, species_order;
  for( size_t species = 0; species < nspecies; ++species )
    if( tracker.species[species].participates and tracker.species[species].has_intra_functional )
      intra_species.push_back( species );
  species_order = intra_species;
  for( size_t species = 0; species < nspecies; ++species )
    if( tracker.species[species].participates and not tracker.species[species].has_intra_functional )
      species_order.push_back( species );
  if( species_order.empty() )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: no participating species");

  const size_t ngrid_tasks = std::distance( task_begin, task_end );

  // ---- 1. Build species task per (grid task, species) with a non-empty shell list, 
  //         with its shells shifted into the union basis
  std::vector<XCTask> species_tasks;            // every species task, in grid-task order
  std::vector<size_t> species_indices;          // per species task, its species
  std::vector<size_t> grid_task_offset;         // per grid task, its first species task
  std::vector<size_t> grid_task_species_count;  // per grid task, how many species tasks it has
  species_tasks.reserve( ngrid_tasks * species_order.size() );
  grid_task_offset.reserve( ngrid_tasks );
  grid_task_species_count.reserve( ngrid_tasks );

  for( auto grid_task = task_begin; grid_task != task_end; ++grid_task ) {
    grid_task_offset.push_back( species_tasks.size() );
    size_t ntasks = 0;
    for( auto species : species_order ) {
      const auto& screening = grid_task->basis_screening( species );
      if( screening.nbe <= 0 ) continue;
      // Build a species task by taking the same grid information from a grid task
      XCTask species_task;
      species_task.iParent       = grid_task->iParent;
      species_task.npts          = static_cast<int32_t>( grid_task->points.size() );
      species_task.dist_nearest  = grid_task->dist_nearest;
      species_task.max_weight    = grid_task->max_weight;
      // Use this species' own screening
      species_task.bfn_screening = screening;
      // Update the shell list to use the union basis's numbering
      for( auto& shell : species_task.bfn_screening.shell_list )
        shell += species_basis_ranges[species].shell_offset;
      species_tasks.push_back( std::move(species_task) );
      species_indices.push_back( species );
      ntasks++;
    }
    grid_task_species_count.push_back( ntasks );
  }

  // Submatrix maps against the union basis
  populate_submat_maps( global_dims.nbf, species_tasks.begin(), species_tasks.end(), basis_map );

  // ---- 2. Device memory of one species task, its grid excluded: the batch
  //         holds one copy of the grid per grid task, counted in step 3
  const bool shared_grid_on_entry = shared_grid_;
  shared_grid_ = true;
  std::vector<size_t> task_bytes( species_tasks.size() );
  for( size_t task = 0; task < species_tasks.size(); ++task )
    task_bytes[task] = get_mem_req( terms, species_tasks[task] )
      + sizeof(double) * size_t( species_tasks[task].npts )   // its weights
      + ( terms.exc_grad ? sizeof(size_t) : 0 );               // its grid offset
  shared_grid_ = shared_grid_on_entry;

  // ---- 3. Greedy batch partition over whole grid tasks, as generate_buffers
  //         partitions over tasks
  if( get_static_mem_requirement() > dynmem_sz )
    GAUXC_GENERIC_EXCEPTION("Insufficient memory to even start!");
  const size_t bytes_available = dynmem_sz - get_static_mem_requirement();

  // Per grid task: one copy of its coordinates, its species tasks, the pair
  // stage's arrays for every pair live on it, and slack for the allocator's
  // alignment. The gradient adds w_times_f, one double per grid point
  const size_t     grid_bytes_per_point  = ( terms.exc_grad ? 4 : 3 ) * sizeof(double);
  constexpr size_t pair_bytes_per_point  = 5 * sizeof(double);
  constexpr size_t alignment_slack_bytes = 32 * 256;

  std::vector<size_t> grid_task_bytes( ngrid_tasks, 0 );
  {
    size_t igrid_task = 0;
    for( auto grid_task = task_begin; grid_task != task_end; ++grid_task, ++igrid_task ) {
      size_t bytes = grid_bytes_per_point * grid_task->points.size();
      for( size_t k = 0; k < grid_task_species_count[igrid_task]; ++k )
        bytes += task_bytes[ grid_task_offset[igrid_task] + k ];
      for( const auto& pair : tracker.pairs ) {
        if( not pair.active ) continue;
        if( grid_task->basis_screening(pair.electron).nbe <= 0 or
            grid_task->basis_screening(pair.particle).nbe <= 0 ) continue;
        bytes += pair_bytes_per_point * grid_task->points.size()
               + sizeof(pair_point_range);
      }
      grid_task_bytes[igrid_task] = bytes + alignment_slack_bytes;
    }
  }

  std::vector<std::pair<size_t,size_t>> grid_task_batches;  // [first, last) grid task of each batch
  {
    size_t igrid_task = 0;
    while( igrid_task < ngrid_tasks ) {
      size_t bytes_left = bytes_available;
      const size_t first = igrid_task;
      while( igrid_task < ngrid_tasks and grid_task_bytes[igrid_task] <= bytes_left ) {
        bytes_left -= grid_task_bytes[igrid_task];
        igrid_task++;
      }
      if( igrid_task == first )
        GAUXC_GENERIC_EXCEPTION("Insufficient device memory for a single multiparticle grid task");
      grid_task_batches.emplace_back( first, igrid_task );
    }
  }

  // ---- 4. Re-order the species tasks within each batch: 
  //         each intra species first, then the rest, each in work ranges of one syr2k size class
  //         (for contiguous memory access during functional evaluation and VXC syr2k launch grid)
  layout.species_tasks.reserve( species_tasks.size() );
  layout.grid_offsets.reserve( species_tasks.size() );
  layout.point_offsets.reserve( species_tasks.size() );
  layout.species_indices.reserve( species_tasks.size() );

  //  TINY    syr2k_tiny
  //  SMALL   cutlass_syr2k_32x32 (threadblock 32x32x16)
  //  GENERAL cutlass_syr2k       (threadblock 64x64x16)
  const syr2k_size_class size_classes[syr2k_size_class_count] = {
    syr2k_size_class::TINY, syr2k_size_class::SMALL, syr2k_size_class::GENERAL };

  for( const auto& grid_task_batch : grid_task_batches ) {
    const size_t first_grid_task = grid_task_batch.first;
    const size_t last_grid_task  = grid_task_batch.second;

    multiparticle_batch batch;
    batch.first_task      = layout.species_tasks.size();
    batch.grid_task_begin = task_begin + first_grid_task;
    batch.grid_task_end   = task_begin + last_grid_task;

    // Where each grid task starts in the batch's one copy of the grid
    std::vector<size_t> grid_task_point_offset( last_grid_task - first_grid_task );
    {
      size_t offset = 0, igrid_in_batch = 0;
      for( auto grid_task = batch.grid_task_begin; grid_task != batch.grid_task_end; ++grid_task, ++igrid_in_batch ) {
        grid_task_point_offset[igrid_in_batch] = offset;
        offset += grid_task->points.size();
      }
      batch.npts_grid = offset;
    }

    // In a batch of grid tasks, group species tasks that belongs to `species` and whose nbe is in `size_class`, 
    // and record that run as one work range. 
    // `species == nspecies` selects every species without an intra functional.
    size_t npts = 0;   // points appended so far: the next task's point offset
    std::vector<size_t> task_point_offset( species_tasks.size(), 0 );
    auto add_work_range = [&]( size_t species, syr2k_size_class size_class ) {
      const size_t range_first = layout.species_tasks.size() - batch.first_task;
      int32_t nbe_max = 0;
      size_t  igrid_in_batch   = 0;
      for( size_t igrid_task = first_grid_task; igrid_task < last_grid_task; ++igrid_task, ++igrid_in_batch ) {
        for( size_t k = 0; k < grid_task_species_count[igrid_task]; ++k ) {
          const size_t task          = grid_task_offset[igrid_task] + k;
          const size_t species_index = species_indices[task];
          const bool   selected      = species == nspecies ? 
                                       not tracker.species[species_index].has_intra_functional : species_index == species;
          if( not selected ) continue;
          const int32_t nbe = species_tasks[task].bfn_screening.nbe;
          if( syr2k_size_class_of(nbe) != size_class ) continue;
          nbe_max = std::max( nbe_max, nbe );
          task_point_offset[task] = npts;
          layout.grid_offsets.push_back( grid_task_point_offset[igrid_in_batch] );
          layout.point_offsets.push_back( npts );
          layout.species_indices.push_back( static_cast<uint32_t>(species_index) );
          npts += size_t( species_tasks[task].npts );
          layout.species_tasks.push_back( std::move(species_tasks[task]) );
        }
      }
      const size_t range_count = layout.species_tasks.size() - batch.first_task - range_first;
      if( range_count )
        batch.task_work_ranges.push_back( { range_first, range_count, nbe_max } );
    };

    batch.intra_point_ranges.assign( nspecies, point_work_range{} );
    // Add possibly 3 size_classes (tiny, small, general) for each intra species (currently electronic only)
    for( size_t species : intra_species ) {
      const size_t first_point = npts;
      for( auto size_class : size_classes ) add_work_range( species, size_class );
      batch.intra_point_ranges[species] = point_work_range{ first_point, npts - first_point };
    }
    batch.ntasks_intra = layout.species_tasks.size() - batch.first_task;
    // Add possibly 3 size_classes (tiny, small, general) for all other species
    for( auto size_class : size_classes ) add_work_range( nspecies, size_class );

    batch.last_task  = layout.species_tasks.size();
    batch.npts_total = npts;

    // The batch's pairs as one point list, ordered by functional, then by pair,
    // then by grid task: a pair's points are one segment of it and a
    // functional's points one range
    std::vector<size_t> pair_order;
    for( size_t pair = 0; pair < tracker.pairs.size(); ++pair )
      if( tracker.pairs[pair].active ) pair_order.push_back( pair );
    std::stable_sort( pair_order.begin(), pair_order.end(),
      [&]( size_t a, size_t b ) {
        return tracker.pairs[a].functional_group < tracker.pairs[b].functional_group;
      });

    size_t npts_pair = 0;
    for( size_t ipair : pair_order ) {
      const auto&  pair        = tracker.pairs[ipair];
      const size_t first_point = npts_pair;
      size_t igrid_in_batch    = 0;
      for( size_t igrid_task = first_grid_task; igrid_task < last_grid_task; ++igrid_task, ++igrid_in_batch ) {
        const size_t not_found     = species_tasks.size();
        size_t       electron_task = not_found;
        size_t       particle_task = not_found;
        for( size_t k = 0; k < grid_task_species_count[igrid_task]; ++k ) {
          const size_t task = grid_task_offset[igrid_task] + k;
          if( species_indices[task] == pair.electron ) electron_task = task;
          if( species_indices[task] == pair.particle ) particle_task = task;
        }
        if( electron_task == not_found or particle_task == not_found ) continue;
        pair_point_range range;
        range.pair              = static_cast<int32_t>( ipair );
        range.npts              = static_cast<int32_t>( (batch.grid_task_begin + igrid_in_batch)->points.size() );
        // The quantum particle's derivative reaches its second channel only
        // when that channel carries the same density
        range.particle_channels = tracker.species[pair.particle].ks_scheme == RKS ? 2 : 1;
        range.first             = npts_pair;
        range.electron_offset   = task_point_offset[electron_task];
        range.particle_offset   = task_point_offset[particle_task];
        range.grid_offset       = grid_task_point_offset[igrid_in_batch];
        batch.pair_point_ranges.push_back( range );
        npts_pair += size_t( range.npts );
      }
      const size_t functional_group = pair.functional_group;
      if( batch.pair_functional_ranges.size() <= functional_group )
        batch.pair_functional_ranges.resize( functional_group + 1 );
      auto& functional_points = batch.pair_functional_ranges[functional_group];
      if( not functional_points.count ) functional_points.first = first_point;
      functional_points.count = npts_pair - functional_points.first;
    }

    layout.batches.push_back( std::move(batch) );
  }

  // The species tasks carry no points of their own; the batch packs the grid once
  for( auto& species_task : layout.species_tasks ) {
    species_task.points.clear();  species_task.points.shrink_to_fit();
    species_task.weights.clear(); species_task.weights.shrink_to_fit();
  }
}


void XCDeviceStackData::generate_buffers_multiparticle(
  integrator_term_tracker terms, const BasisSetMap& basis_map,
  multiparticle_layout& layout, size_t ibatch ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  auto& batch = layout.batches.at(ibatch);
  auto batch_tasks_begin = layout.species_tasks.begin() + batch.first_task;
  auto batch_tasks_end   = layout.species_tasks.begin() + batch.last_task;

  grid_npts_    = batch.npts_grid;
  ntasks_intra_ = batch.ntasks_intra;
  task_grid_offset_.assign( layout.grid_offsets.begin() + batch.first_task,
                               layout.grid_offsets.begin() + batch.last_task );
  task_work_ranges_ = batch.task_work_ranges;

  // ---- 1. The batch's single copy of the grid, ahead of everything else
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );
  base_stack.points_x_device = mem.aligned_alloc<double>( batch.npts_grid, 256, csl );
  base_stack.points_y_device = mem.aligned_alloc<double>( batch.npts_grid, 256, csl );
  base_stack.points_z_device = mem.aligned_alloc<double>( batch.npts_grid, 256, csl );
  // The weights are per species-task point: scale_*_output, inc_exc and
  // inc_nel read one array over the batch's species tasks
  base_stack.weights_device  = mem.aligned_alloc<double>( batch.npts_total, 256, csl );

  // ---- 2. Everything else is sized by the species-task list, through the
  //         per-task allocator with the grid excluded
  const bool shared_grid_on_entry = shared_grid_;
  shared_grid_ = true;
  auto buf = allocate_dynamic_stack( terms, batch_tasks_begin, batch_tasks_end,
    device_buffer_t{ mem.stack(), mem.nleft() } );

  // vrho and vgamma start at zero: the intra XC writes its own species' points only
  {
    const size_t npts = total_npts_task_batch;
    auto zero = [&]( double* array, const char* msg ) {
      if( array ) device_backend_->set_zero_async_master_queue( npts, array, msg );
    };
    if( terms.ks_scheme == RKS ) {
      zero( base_stack.vrho_eval_device, "MultiParticle vrho Zero" );
      if( terms.xc_approx == GGA )
        zero( base_stack.vgamma_eval_device, "MultiParticle vgamma Zero" );
    } else {
      zero( base_stack.vrho_pos_eval_device, "MultiParticle vrho+ Zero" );
      zero( base_stack.vrho_neg_eval_device, "MultiParticle vrho- Zero" );
      if( terms.xc_approx == GGA ) {
        zero( base_stack.vgamma_pp_eval_device, "MultiParticle vgamma++ Zero" );
        zero( base_stack.vgamma_pm_eval_device, "MultiParticle vgamma+- Zero" );
        zero( base_stack.vgamma_mm_eval_device, "MultiParticle vgamma-- Zero" );
      }
    }
  }

  // ---- 3. The pair stage's arrays, from what the species-task list left
  pair_stack.reset();
  {
    size_t npts_pair = 0;
    for( const auto& range : batch.pair_point_ranges ) npts_pair += size_t( range.npts );
    if( npts_pair ) {
      auto [ pair_ptr, pair_bytes ] = buf;
      buffer_adaptor pair_mem( pair_ptr, pair_bytes );
      pair_stack.density_device = pair_mem.aligned_alloc<double>( 2*npts_pair, 256, csl );
      pair_stack.eps_device     = pair_mem.aligned_alloc<double>( npts_pair, 256, csl );
      pair_stack.vrho_device    = pair_mem.aligned_alloc<double>( 2*npts_pair, 256, csl );
      pair_stack.ranges_device  = pair_mem.aligned_alloc<pair_point_range>(
        batch.pair_point_ranges.size(), 256, csl );
      buf = device_buffer_t{ pair_mem.stack(), pair_mem.nleft() };
      pair_stack.npts    = npts_pair;
      pair_stack.nranges = batch.pair_point_ranges.size();
    }
  }

  // ---- 3b. EXC gradient: the task grid offsets and w_times_f
  if( terms.exc_grad ) {
    auto [ grad_ptr, grad_bytes ] = buf;
    buffer_adaptor grad_mem( grad_ptr, grad_bytes );
    task_grid_offset_device_ = grad_mem.aligned_alloc<size_t>(
      task_grid_offset_.size(), 256, csl );
    w_times_f_device_        = grad_mem.aligned_alloc<double>( batch.npts_grid, 256, csl );
    buf = device_buffer_t{ grad_mem.stack(), grad_mem.nleft() };
  }

  // ---- 4. Pack: the grid and the pair point ranges here, once; the species
  //         tasks through the per-task packer, which skips the grid
  {
    std::vector<double> points_x, points_y, points_z, grid_weights, weights;
    points_x.reserve( batch.npts_grid );
    points_y.reserve( batch.npts_grid );
    points_z.reserve( batch.npts_grid );
    grid_weights.reserve( batch.npts_grid );
    weights.reserve( batch.npts_total );
    for( auto grid_task = batch.grid_task_begin; grid_task != batch.grid_task_end; ++grid_task ) {
      for( const auto& point : grid_task->points ) {
        points_x.push_back( point[0] );
        points_y.push_back( point[1] );
        points_z.push_back( point[2] );
      }
      grid_weights.insert( grid_weights.end(), grid_task->weights.begin(), grid_task->weights.end() );
    }
    // Each species task repeats its grid task's weights
    for( size_t task = batch.first_task; task < batch.last_task; ++task ) {
      const size_t grid_offset = layout.grid_offsets[task];
      const size_t npts        = size_t( layout.species_tasks[task].npts );
      weights.insert( weights.end(), grid_weights.begin() + grid_offset,
        grid_weights.begin() + grid_offset + npts );
    }
    if( points_x.size() != batch.npts_grid or weights.size() != batch.npts_total )
      GAUXC_GENERIC_EXCEPTION("MultiParticle: inconsistent grid allocation");
    device_backend_->copy_async( batch.npts_grid,  points_x.data(), base_stack.points_x_device, "multiparticle points_x" );
    device_backend_->copy_async( batch.npts_grid,  points_y.data(), base_stack.points_y_device, "multiparticle points_y" );
    device_backend_->copy_async( batch.npts_grid,  points_z.data(), base_stack.points_z_device, "multiparticle points_z" );
    device_backend_->copy_async( batch.npts_total, weights.data(),  base_stack.weights_device,  "multiparticle weights"  );
    if( pair_stack.nranges )
      device_backend_->copy_async( pair_stack.nranges, batch.pair_point_ranges.data(),
        pair_stack.ranges_device, "multiparticle pair ranges" );
    if( task_grid_offset_device_ )
      device_backend_->copy_async( task_grid_offset_.size(), task_grid_offset_.data(),
        task_grid_offset_device_, "multiparticle task grid offsets" );
    device_backend_->master_queue_synchronize();
  }

  pack_and_send( terms, batch_tasks_begin, batch_tasks_end, basis_map );
  shared_grid_ = shared_grid_on_entry;
}


void XCDeviceStackData::allocate_static_data_multiparticle( size_t nspecies,
  size_t npairs ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  nspecies_ = nspecies;
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );
  multiparticle_static.intra_exc_device = mem.aligned_alloc<double>( nspecies, csl );
  if( npairs )
    multiparticle_static.pair_exc_device = mem.aligned_alloc<double>( npairs, csl );
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft();

  device_backend_->set_zero( nspecies, multiparticle_static.intra_exc_device, "MultiParticle intra EXC Zero" );
  if( npairs )
    device_backend_->set_zero( npairs, multiparticle_static.pair_exc_device, "MultiParticle pair EXC Zero" );
}


void XCDeviceStackData::retrieve_intra_exc( double* intra_EXC, size_t nspecies ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  if( not multiparticle_static.intra_exc_device )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: intra EXC accumulator not allocated");
  device_backend_->copy_async( nspecies, multiparticle_static.intra_exc_device,
    intra_EXC, "MultiParticle intra EXC D2H" );
  device_backend_->master_queue_synchronize();
}


void XCDeviceStackData::retrieve_pair_exc( double* pair_EXC, size_t npairs ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  if( not npairs ) return;
  if( not multiparticle_static.pair_exc_device )
    GAUXC_GENERIC_EXCEPTION("MultiParticle: pair EXC accumulator not allocated");
  device_backend_->copy_async( npairs, multiparticle_static.pair_exc_device,
    pair_EXC, "MultiParticle pair EXC D2H" );
  device_backend_->master_queue_synchronize();
}


double* XCDeviceStackData::intra_exc_device_data( size_t species ) {
  return multiparticle_static.intra_exc_device + species;
}


}

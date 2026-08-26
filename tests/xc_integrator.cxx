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
#include "ut_common.hpp"
#include <gauxc/xc_integrator.hpp>
#include <gauxc/xc_integrator/impl.hpp>
#include <gauxc/xc_integrator/integrator_factory.hpp>
#include <gauxc/molecular_weights.hpp>

#include <gauxc/molgrid/defaults.hpp>

#include <gauxc/external/hdf5.hpp>
#include <highfive/H5File.hpp>
#include <Eigen/Core>

using namespace GauXC;


void test_xc_integrator( ExecutionSpace ex, const RuntimeEnvironment& rt,
  std::string reference_file, 
  functional_type& func, 
  PruningScheme pruning_scheme,
  bool check_grad,
  bool check_integrate_den,
  bool check_k,
  std::string integrator_kernel = "Default",  
  std::string reduction_kernel  = "Default",
  std::string lwd_kernel        = "Default") {

  // Read the reference file
  using matrix_type = Eigen::MatrixXd;
  Molecule mol;
  BasisSet<double> basis;
  matrix_type P, Pz, Py, Px, VXC_ref, VXCz_ref, VXCy_ref, VXCx_ref, K_ref;
  double EXC_ref;
  std::vector<double> EXC_GRAD_ref_HellFey, EXC_GRAD_ref_Full;
  bool has_k = false, has_exc_grad_HellFey = false, has_exc_grad_full = false, rks = true, uks = false, gks = false;
  {
    read_hdf5_record( mol,   reference_file, "/MOLECULE" );
    read_hdf5_record( basis, reference_file, "/BASIS"    );

    HighFive::File file( reference_file, HighFive::File::ReadOnly );
    
    std::string den="/DENSITY";
    std::string den2="/DENSITY_Z";
    std::string den3="/DENSITY_Y";
    std::string den4="/DENSITY_X";
    std::string vxc="/VXC";
    std::string vxc2="VXC_Z";
    std::string vxc3="VXC_Y";
    std::string vxc4="VXC_X";

    if (file.exist("/DENSITY_Z")) { rks = false; }

    if (file.exist("/DENSITY_Z") and not file.exist("/DENSITY_Y") and not file.exist("/DENSITY_X")) {
       den="/DENSITY_SCALAR";
       vxc="/VXC_SCALAR";
       uks=true;
    }
     
    if (file.exist("/DENSITY_X") and file.exist("/DENSITY_Y") and file.exist("/DENSITY_Z")) {
       den="/DENSITY_SCALAR";
       vxc="/VXC_SCALAR";
       gks=true;
    }
 
    auto dset = file.getDataSet(den);
    
    auto dims = dset.getDimensions();
    P        = matrix_type( dims[0], dims[1] );
    VXC_ref  = matrix_type( dims[0], dims[1] );
    if (not rks) {
      Pz       = matrix_type( dims[0], dims[1] );
      VXCz_ref = matrix_type( dims[0], dims[1] );
    } 
    if (gks) {
      Py       = matrix_type( dims[0], dims[1] );
      VXCy_ref = matrix_type( dims[0], dims[1] );
      Px       = matrix_type( dims[0], dims[1] );
      VXCx_ref = matrix_type( dims[0], dims[1] );
    }


    dset.read( P.data() );
    dset = file.getDataSet(vxc);
    dset.read( VXC_ref.data() );

    if (not rks) {
      dset = file.getDataSet(den2);
      dset.read( Pz.data() );
      dset = file.getDataSet(vxc2);
      dset.read( VXCz_ref.data() );
    }

    if (gks) {
      dset = file.getDataSet(den3);
      dset.read( Py.data() );
      dset = file.getDataSet(vxc3);
      dset.read( VXCy_ref.data() );
      dset = file.getDataSet(den4);
      dset.read( Px.data() );
      dset = file.getDataSet(vxc4);
      dset.read( VXCx_ref.data() );
    }    

    dset = file.getDataSet("/EXC");
    dset.read( &EXC_ref );

    // Check for new unified /EXC_GRAD dataset with attribute
    if( file.exist("/EXC_GRAD") ) {
      dset = file.getDataSet("/EXC_GRAD");
      EXC_GRAD_ref_Full.resize( 3*mol.size() );
      
      // Check for attribute indicating whether weight derivatives are included
      bool exc_grad_includes_weight_derivatives = false; // Default to Hellmann-Feynman
      try {
        auto attr = dset.getAttribute("includes_weight_derivatives");
        int attr_value;
        attr.read( attr_value );
        exc_grad_includes_weight_derivatives = (attr_value != 0);
      } catch(... ) { }
      
      if( exc_grad_includes_weight_derivatives ) {
        dset.read( EXC_GRAD_ref_Full.data() );
        has_exc_grad_full = true;
      } else {
        dset.read( EXC_GRAD_ref_HellFey.data() );
        has_exc_grad_HellFey = true;
      }
    }
    // Check for other type of EXC_GRAD
    if( file.exist("/EXC_GRAD_HELLFEY") and not has_exc_grad_HellFey ) {
      EXC_GRAD_ref_HellFey.resize( 3*mol.size() );
      dset = file.getDataSet("/EXC_GRAD_HELLFEY");
      dset.read( EXC_GRAD_ref_HellFey.data() );
      has_exc_grad_HellFey = true;
    }
    if( file.exist("/EXC_GRAD_FULL") and not has_exc_grad_full ) {
      EXC_GRAD_ref_Full.resize( 3*mol.size() );
      dset = file.getDataSet("/EXC_GRAD_FULL");
      dset.read( EXC_GRAD_ref_Full.data() );
      has_exc_grad_full = true;
    }
    
    has_k = file.exist("/K");
    if(has_k) {
        K_ref = matrix_type(dims[0], dims[1]);
        dset = file.getDataSet("/K");
        dset.read( K_ref.data() );
    }
  }

  if( gks and ex == ExecutionSpace::Device and func.is_mgga() ) return;

  for( auto& sh : basis ) 
    sh.set_shell_tolerance( std::numeric_limits<double>::epsilon() );

  auto mg = MolGridFactory::create_default_molgrid(mol, pruning_scheme,
    BatchSize(512), RadialQuad::MuraKnowles, AtomicGridSizeDefault::UltraFineGrid);

  // Construct Load Balancer
  LoadBalancerFactory lb_factory(ExecutionSpace::Host, "Default");
  auto lb = lb_factory.get_instance(rt, mol, mg, basis);

  // Construct Weights Module
  MolecularWeightsFactory mw_factory( ex, "Default", MolecularWeightsSettings{} );
  auto mw = mw_factory.get_instance();

  // Apply partition weights
  mw.modify_weights(lb);

  // Construct XC Functional
  //auto Spin = uks ? ExchCXX::Spin::Polarized : ExchCXX::Spin::Unpolarized;
  //functional_type func( ExchCXX::Backend::builtin, func_key, Spin );

  // Construct XCIntegrator
  XCIntegratorFactory<matrix_type> integrator_factory( ex, "Replicated", 
    integrator_kernel, lwd_kernel, reduction_kernel );
  auto integrator = integrator_factory.get_instance( func, lb );

  // Integrate Density
  if( check_integrate_den and rks) {
    auto N_EL_ref = std::accumulate( mol.begin(), mol.end(), size_t{0},
      [](const auto& a, const auto &b) { return a + b.Z.get(); });
    auto N_EL = integrator.integrate_den( P );
    // Factor of 2 b/c P is the alpha density for RKS
    CHECK( N_EL == Approx(N_EL_ref/2.0).epsilon(1e-6) );
  }

  // Integrate EXC/VXC
  if ( rks ) {
    auto [ EXC, VXC ] = integrator.eval_exc_vxc( P );

    // Check EXC/VXC
    auto VXC_diff_nrm = ( VXC - VXC_ref ).norm();
    CHECK( EXC == Approx( EXC_ref ) );
    CHECK( VXC_diff_nrm / basis.nbf() < 1e-10 ); 
    // Check if the integrator propagates state correctly
    {
      auto [ EXC1, VXC1 ] = integrator.eval_exc_vxc( P );
      CHECK( EXC1 == Approx( EXC_ref ) );
      auto VXC1_diff_nrm = ( VXC1 - VXC_ref ).norm();
      CHECK( VXC1_diff_nrm / basis.nbf() < 1e-10 ); 
    }

    // Check EXC-only path
    auto EXC2 = integrator.eval_exc( P );
    CHECK(EXC2 == Approx(EXC));

  } else if (uks) {
    auto [ EXC, VXC, VXCz ] = integrator.eval_exc_vxc( P, Pz );

    // Check EXC/VXC
    auto VXC_diff_nrm = ( VXC - VXC_ref ).norm();
    auto VXCz_diff_nrm = ( VXCz - VXCz_ref ).norm();
    CHECK( EXC == Approx( EXC_ref ) );
    CHECK( VXC_diff_nrm / basis.nbf() < 1e-10 );
    CHECK( VXCz_diff_nrm / basis.nbf() < 1e-10 );
    // Check if the integrator propagates state correctly
    {
      auto [ EXC1, VXC1, VXCz1 ] = integrator.eval_exc_vxc( P, Pz );
      CHECK( EXC1 == Approx( EXC_ref ) );
      auto VXC1_diff_nrm = ( VXC1 - VXC_ref ).norm();
      auto VXCz1_diff_nrm = ( VXCz1 - VXCz_ref ).norm();
      CHECK( VXC1_diff_nrm / basis.nbf() < 1e-10 );
      CHECK( VXCz1_diff_nrm / basis.nbf() < 1e-10 );
    }

    // Check EXC-only path
    auto EXC2 = integrator.eval_exc( P, Pz );
    CHECK(EXC2 == Approx(EXC));
  } else if (gks) {
    auto [ EXC, VXC, VXCz, VXCy, VXCx ] = integrator.eval_exc_vxc( P, Pz, Py, Px );

    // Check EXC/VXC
    auto VXC_diff_nrm = ( VXC - VXC_ref ).norm();
    auto VXCz_diff_nrm = ( VXCz - VXCz_ref ).norm();
    auto VXCy_diff_nrm = ( VXCy - VXCy_ref ).norm();
    auto VXCx_diff_nrm = ( VXCx - VXCx_ref ).norm();

    CHECK( EXC == Approx( EXC_ref ) );
    CHECK( VXC_diff_nrm / basis.nbf() < 1e-10 );
    CHECK( VXCz_diff_nrm / basis.nbf() < 1e-10 );
    CHECK( VXCy_diff_nrm / basis.nbf() < 1e-10 );
    CHECK( VXCx_diff_nrm / basis.nbf() < 1e-10 );
    // Check if the integrator propagates state correctly
    {
      auto [ EXC1, VXC1, VXCz1, VXCy1, VXCx1] = integrator.eval_exc_vxc( P, Pz, Py, Px );
      CHECK( EXC1 == Approx( EXC_ref ) );
      auto VXC1_diff_nrm = ( VXC1 - VXC_ref ).norm();
      auto VXCz1_diff_nrm = ( VXCz1 - VXCz_ref ).norm();
      auto VXCy1_diff_nrm = ( VXCy1 - VXCy_ref ).norm();
      auto VXCx1_diff_nrm = ( VXCx1 - VXCx_ref ).norm();
      CHECK( VXC1_diff_nrm / basis.nbf() < 1e-10 );
      CHECK( VXCz1_diff_nrm / basis.nbf() < 1e-10 );
      CHECK( VXCy1_diff_nrm / basis.nbf() < 1e-10 );
      CHECK( VXCx1_diff_nrm / basis.nbf() < 1e-10 );
    }

    // Check EXC-only path
    auto EXC2 = integrator.eval_exc( P, Pz, Py, Px );
    CHECK(EXC2 == Approx(EXC));
  }



  // Check EXC Grad
  if( check_grad and has_exc_grad_full ) {
    IntegratorSettingsEXC_GRAD exc_grad_settings;
    exc_grad_settings.include_weight_derivatives = true; // Use full gradient (default)
    auto EXC_GRAD = rks ? integrator.eval_exc_grad( P, exc_grad_settings ) : integrator.eval_exc_grad( P, Pz, exc_grad_settings );
    using map_type = Eigen::Map<Eigen::MatrixXd>;
    map_type EXC_GRAD_ref_map( EXC_GRAD_ref_Full.data(), mol.size(), 3 );
    map_type EXC_GRAD_map( EXC_GRAD.data(), mol.size(), 3 );
    auto EXC_GRAD_diff_nrm = (EXC_GRAD_ref_map - EXC_GRAD_map).norm();
    INFO("comparing full gradient");
    CHECK( EXC_GRAD_diff_nrm / std::sqrt(3.0*mol.size()) < 1e-8 );
  }
  if( check_grad and has_exc_grad_HellFey ) {
    IntegratorSettingsEXC_GRAD exc_grad_settings;
    exc_grad_settings.include_weight_derivatives = false; // Use Hellmann-Feynman gradient
    auto EXC_GRAD = rks ? integrator.eval_exc_grad( P, exc_grad_settings ) : integrator.eval_exc_grad( P, Pz, exc_grad_settings );
    using map_type = Eigen::Map<Eigen::MatrixXd>;
    map_type EXC_GRAD_ref_map( EXC_GRAD_ref_HellFey.data(), mol.size(), 3 );
    map_type EXC_GRAD_map( EXC_GRAD.data(), mol.size(), 3 );
    auto EXC_GRAD_diff_nrm = (EXC_GRAD_ref_map - EXC_GRAD_map).norm();
    INFO("comparing Hellmann-Feynman gradient");
    CHECK( EXC_GRAD_diff_nrm / std::sqrt(3.0*mol.size()) < 1e-8 );
  }


  // Check K
  if( has_k and check_k and rks ) {
    auto max_l = basis.max_l();
    if(max_l > 2 and ex == ExecutionSpace::Device) {
      std::cout << "Skiping device sn-K + L > 2" << std::endl;
      return;
    }
    auto K = integrator.eval_exx( P );
    CHECK((K - K.transpose()).norm() < std::numeric_limits<double>::epsilon()); // Symmetric
    CHECK( (K - K_ref).norm() / basis.nbf() < 1e-7 );
  }

}

void test_integrator(std::string reference_file, functional_type& func, PruningScheme pruning_scheme) {

#ifdef GAUXC_HAS_DEVICE
  auto rt = DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.9);
#else
  auto rt = RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
#endif

#ifdef GAUXC_HAS_HOST
    SECTION( "Host" ) {
      SECTION("Reference") {
        test_xc_integrator( ExecutionSpace::Host, rt, reference_file, func,
          pruning_scheme, true, true, true );
      }
      SECTION("ShellBatched") {
        test_xc_integrator( ExecutionSpace::Host, rt, reference_file, func,
          pruning_scheme, false, false, false, "ShellBatched" );
      }
    }
#endif

#ifdef GAUXC_HAS_DEVICE
  SECTION( "Device" ) {
    bool check_grad = true;
    bool check_k    = true;
    #ifdef GAUXC_HAS_HIP
    check_grad = false;
    check_k    = false;
    #endif
    SECTION( "Incore - MPI Reduction" ) {
      test_xc_integrator( ExecutionSpace::Device, rt,
        reference_file, func, pruning_scheme,  
        check_grad, true, check_k, "Default" );
    }

    #ifdef GAUXC_HAS_MAGMA
    SECTION( "Incore - MPI Reduction - MAGMA" ) {
      if(not func.is_mgga() and not func.is_polarized()) {
        test_xc_integrator( ExecutionSpace::Device, rt,
          reference_file, func, pruning_scheme,
          false, true, check_k, "Default", "Default", 
          "Scheme1-MAGMA" );
      }
    }
    #endif

    #ifdef GAUXC_HAS_CUTLASS
    SECTION( "Incore - MPI Reduction - CUTLASS" ) {
      test_xc_integrator( ExecutionSpace::Device, rt, 
        reference_file, func, pruning_scheme,
        true, true, false, "Default", "Default", 
        "Scheme1-CUTLASS" );
    }
    #endif


    #ifdef GAUXC_HAS_NCCL
    SECTION( "Incore - NCCL Reduction" ) {
      test_xc_integrator( ExecutionSpace::Device, rt,
        reference_file, func, pruning_scheme, 
        false, false, false, "Default", "NCCL" );
    }
    #endif

    // SECTION( "ShellBatched" ) {
    //   test_xc_integrator( ExecutionSpace::Device, rt, 
    //     reference_file, func, pruning_scheme,  
    //     false, false, false, "ShellBatched" );
    // }
  }
#endif

}

functional_type make_functional(ExchCXX::Functional func_key, ExchCXX::Spin spin) {
  return functional_type(ExchCXX::Backend::builtin, func_key, spin);
}


// ----------------------------------------------------------------------------
// NEO (multiparticle) XC integrator
//
// A NEO reference file reuses the single-species dataset names for the
// electronic species (which is always species 0, as the EPC kernels require
// the electron density first) and adds flat, indexed datasets for each
// additional (protonic) species i, i.e. GauXC species i+1:
//
//   /PROTONIC_BASIS_<i>            /PROTONIC_VXC_SCALAR_<i>
//   /PROTONIC_DENSITY_SCALAR_<i>   /PROTONIC_VXC_Z_<i>
//   /PROTONIC_DENSITY_Z_<i>        /PROTONIC_EXC_<i>
//   /EPC_EXC_<j>    inter-species energy of pair j = (electron, species j+1)
//
// ----------------------------------------------------------------------------

using mp_density_type = XCIntegrator<Eigen::MatrixXd>::multiparticle_density;

std::string neo_dset( std::string base, size_t i ) {
  return base + std::to_string(i);
}

void test_neo_xc_integrator( ExecutionSpace ex, const RuntimeEnvironment& rt,
  std::string reference_file,
  functional_type& func,      // electronic (intra) functional
  functional_type& epc_func,  // electron/particle (inter, EPC) functional
  PruningScheme pruning_scheme,
  bool check_grad ) {

  using matrix_type = Eigen::MatrixXd;
  Molecule mol;
  BasisSet<double> basis;
  std::vector<BasisSet<double>> prot_basis;
  matrix_type P, Pz, VXC_ref, VXCz_ref;
  std::vector<matrix_type> prot_Ps, prot_Pz, prot_VXCs_ref, prot_VXCz_ref;
  std::vector<double> prot_EXC_ref, EPC_EXC_ref, EXC_GRAD_ref;
  double EXC_ref = 0.;
  size_t nprot = 0;
  bool uks = false, has_exc_grad = false;
  std::vector<bool> prot_uks;

  {
    HighFive::File file( reference_file, HighFive::File::ReadOnly );
    uks          = file.exist("/DENSITY_Z");
    has_exc_grad = file.exist("/EXC_GRAD");
    while( file.exist( neo_dset("/PROTONIC_BASIS_", nprot) ) ) nprot++;
  }
  REQUIRE( nprot > 0 );
  const size_t nspecies = nprot + 1;

  read_hdf5_record( mol,   reference_file, "/MOLECULE" );
  read_hdf5_record( basis, reference_file, "/BASIS"    );
  prot_basis.resize( nprot );
  for( size_t i = 0; i < nprot; ++i )
    read_hdf5_record( prot_basis[i], reference_file,
      neo_dset("/PROTONIC_BASIS_", i) );

  {
    HighFive::File file( reference_file, HighFive::File::ReadOnly );

    std::string den = "/DENSITY", vxc = "/VXC";
    if( uks ) { den = "/DENSITY_SCALAR"; vxc = "/VXC_SCALAR"; }

    auto dset = file.getDataSet(den);
    auto dims = dset.getDimensions();
    P       = matrix_type( dims[0], dims[1] );
    VXC_ref = matrix_type( dims[0], dims[1] );
    dset.read( P.data() );
    dset = file.getDataSet(vxc);
    dset.read( VXC_ref.data() );
    if( uks ) {
      Pz       = matrix_type( dims[0], dims[1] );
      VXCz_ref = matrix_type( dims[0], dims[1] );
      dset = file.getDataSet("/DENSITY_Z"); dset.read( Pz.data()       );
      dset = file.getDataSet("/VXC_Z"    ); dset.read( VXCz_ref.data() );
    }
    dset = file.getDataSet("/EXC");
    dset.read( &EXC_ref );

    prot_Ps.resize(nprot); prot_Pz.resize(nprot);
    prot_VXCs_ref.resize(nprot); prot_VXCz_ref.resize(nprot);
    prot_EXC_ref.resize(nprot); EPC_EXC_ref.resize(nprot);
    prot_uks.assign(nprot, false);
    for( size_t i = 0; i < nprot; ++i ) {
      dset = file.getDataSet( neo_dset("/PROTONIC_DENSITY_SCALAR_", i) );
      auto pdims = dset.getDimensions();
      prot_Ps[i]       = matrix_type( pdims[0], pdims[1] );
      prot_VXCs_ref[i] = matrix_type( pdims[0], pdims[1] );
      dset.read( prot_Ps[i].data() );
      dset = file.getDataSet( neo_dset("/PROTONIC_VXC_SCALAR_", i) );
      dset.read( prot_VXCs_ref[i].data() );

      prot_uks[i] = file.exist( neo_dset("/PROTONIC_DENSITY_Z_", i) );
      if( prot_uks[i] ) {
        prot_Pz[i]       = matrix_type( pdims[0], pdims[1] );
        prot_VXCz_ref[i] = matrix_type( pdims[0], pdims[1] );
        dset = file.getDataSet( neo_dset("/PROTONIC_DENSITY_Z_", i) );
        dset.read( prot_Pz[i].data() );
        dset = file.getDataSet( neo_dset("/PROTONIC_VXC_Z_", i) );
        dset.read( prot_VXCz_ref[i].data() );
      }

      dset = file.getDataSet( neo_dset("/PROTONIC_EXC_", i) );
      dset.read( &prot_EXC_ref[i] );
      dset = file.getDataSet( neo_dset("/EPC_EXC_", i) );
      dset.read( &EPC_EXC_ref[i] );
    }

    if( has_exc_grad ) {
      EXC_GRAD_ref.resize( 3*mol.size() );
      dset = file.getDataSet("/EXC_GRAD");
      dset.read( EXC_GRAD_ref.data() );
    }
  }

  for( auto& sh : basis )
    sh.set_shell_tolerance( std::numeric_limits<double>::epsilon() );
  for( auto& b : prot_basis ) for( auto& sh : b )
    sh.set_shell_tolerance( std::numeric_limits<double>::epsilon() );

  auto mg = MolGridFactory::create_default_molgrid(mol, pruning_scheme,
    BatchSize(512), RadialQuad::MuraKnowles, AtomicGridSizeDefault::UltraFineGrid);

  // Construct multi-basis Load Balancer (species 0 == electron)
  std::vector< BasisSet<double> > bases;
  bases.reserve( nspecies );
  bases.push_back( basis );
  for( const auto& b : prot_basis ) bases.push_back( b );

  LoadBalancerFactory lb_factory(ExecutionSpace::Host, "Default");
  auto lb = lb_factory.get_instance(rt, mol, mg, bases);

  // Construct Weights Module
  MolecularWeightsFactory mw_factory( ex, "Default", MolecularWeightsSettings{} );
  auto mw = mw_factory.get_instance();
  mw.modify_weights(lb);

  // Construct the multiparticle functional spec: an intra functional for the
  // electron only, and one EPC pair per protonic species.
  MultiParticleFunctionalSpec spec;
  spec.intra_functionals.resize( nspecies );
  spec.intra_functionals[0].push_back( std::make_shared<functional_type>(func) );
  for( size_t i = 0; i < nprot; ++i )
    spec.inter_functionals.push_back( MultiParticlePairFunctional{ 0, i+1,
      { std::make_shared<functional_type>(epc_func) } } );

  MultiParticleXCTerms terms;
  for( size_t p = 0; p < nspecies; ++p ) {
    terms.active_intra.push_back(p);
    terms.vxc_targets.push_back(p);
  }
  for( size_t j = 0; j < nprot; ++j ) terms.active_inter.push_back(j);

  // Construct XCIntegrator
  XCIntegratorFactory<matrix_type> integrator_factory( ex, "Replicated",
    "Default", "Default", "Default" );
  auto integrator = integrator_factory.get_instance( func, lb );

  std::vector<mp_density_type> densities;
  densities.push_back( mp_density_type{ &P, uks ? &Pz : nullptr } );
  for( size_t i = 0; i < nprot; ++i )
    densities.push_back( mp_density_type{ &prot_Ps[i],
      prot_uks[i] ? &prot_Pz[i] : nullptr } );

  auto exc_approx = []( double x ) { return Approx(x).epsilon(1e-10).margin(1e-14); };

  auto result = integrator.eval_exc_vxc( densities, spec, terms );

  // Structure
  REQUIRE( lb.basis_count()             == nspecies );
  REQUIRE( result.intra_exc.size()      == nspecies );
  REQUIRE( result.VXCs.size()           == nspecies );
  REQUIRE( result.VXCz.size()           == nspecies );
  REQUIRE( result.inter_pair_exc.size() == nprot    );

  // Per-species intra EXC
  CHECK( result.intra_exc[0] == exc_approx(EXC_ref) );
  for( size_t i = 0; i < nprot; ++i ) {
    INFO( "protonic species " << i );
    CHECK( result.intra_exc[i+1] == exc_approx(prot_EXC_ref[i]) );
  }

  // Per-pair EPC EXC, each pair separately and their sum
  double epc_sum = 0.;
  for( size_t j = 0; j < nprot; ++j ) {
    INFO( "EPC pair " << j );
    CHECK( result.inter_pair_exc[j] == exc_approx(EPC_EXC_ref[j]) );
    epc_sum += result.inter_pair_exc[j];
  }
  CHECK( result.inter_exc == exc_approx(epc_sum) );

  // Per-species, per-channel VXC
  CHECK( (result.VXCs[0] - VXC_ref).norm() / basis.nbf() < 1e-10 );
  if( uks ) CHECK( (result.VXCz[0] - VXCz_ref).norm() / basis.nbf() < 1e-10 );
  for( size_t i = 0; i < nprot; ++i ) {
    INFO( "protonic species " << i );
    const auto pnbf = prot_basis[i].nbf();
    CHECK( (result.VXCs[i+1] - prot_VXCs_ref[i]).norm() / pnbf < 1e-10 );
    if( prot_uks[i] )
      CHECK( (result.VXCz[i+1] - prot_VXCz_ref[i]).norm() / pnbf < 1e-10 );
  }

  // Every returned VXC is symmetric
  for( size_t p = 0; p < nspecies; ++p ) {
    INFO( "species " << p );
    const auto pnbf = bases[p].nbf();
    if( result.VXCs[p].size() )
      CHECK( (result.VXCs[p] - result.VXCs[p].transpose()).norm() / pnbf < 1e-12 );
    if( result.VXCz[p].size() )
      CHECK( (result.VXCz[p] - result.VXCz[p].transpose()).norm() / pnbf < 1e-12 );
  }

  // A high-spin quantum particle has Ps == Pz. This is a property of the
  // converged density rather than of GauXC, and it is what a future device
  // alpha-only particle channel would exploit, so it is pinned here.
  for( size_t i = 0; i < nprot; ++i ) if( prot_uks[i] ) {
    INFO( "protonic species " << i );
    CHECK( (prot_Ps[i] - prot_Pz[i]).norm() / prot_basis[i].nbf() < 1e-12 );
  }

  // Check if the integrator propagates state correctly. This matters more than
  // in the single-species case: the multiparticle local work re-sorts the load
  // balancer's task vector in place on every call.
  {
    auto r = integrator.eval_exc_vxc( densities, spec, terms );
    CHECK( r.intra_exc[0] == exc_approx(EXC_ref) );
    CHECK( (r.VXCs[0] - VXC_ref).norm() / basis.nbf() < 1e-10 );
    for( size_t i = 0; i < nprot; ++i ) {
      INFO( "protonic species " << i );
      CHECK( r.intra_exc[i+1]      == exc_approx(prot_EXC_ref[i]) );
      CHECK( r.inter_pair_exc[i]   == exc_approx(EPC_EXC_ref[i])  );
      CHECK( (r.VXCs[i+1] - prot_VXCs_ref[i]).norm() / prot_basis[i].nbf() < 1e-10 );
    }
  }

  // Non-target species come back as empty matrices, and dropping a VXC target
  // must not perturb any energy
  {
    auto t = terms; t.vxc_targets = { 0 };
    auto r = integrator.eval_exc_vxc( densities, spec, t );
    CHECK( (r.VXCs[0] - result.VXCs[0]).norm() / basis.nbf() < 1e-12 );
    if( not uks ) CHECK( r.VXCz[0].size() == 0 );
    for( size_t i = 0; i < nprot; ++i ) {
      INFO( "protonic species " << i );
      CHECK( r.VXCs[i+1].size() == 0 );
      CHECK( r.VXCz[i+1].size() == 0 );
      CHECK( r.intra_exc[i+1]    == exc_approx(result.intra_exc[i+1])    );
      CHECK( r.inter_pair_exc[i] == exc_approx(result.inter_pair_exc[i]) );
    }
    CHECK( r.intra_exc[0] == exc_approx(result.intra_exc[0]) );
  }

  // Check EXC-only path. There is no separate multiparticle eval_exc; an empty
  // vxc_targets is its analogue and it disables the EPC potential scatter.
  {
    auto t = terms; t.vxc_targets.clear();
    auto r = integrator.eval_exc_vxc( densities, spec, t );
    for( size_t p = 0; p < nspecies; ++p ) {
      INFO( "species " << p );
      CHECK( r.intra_exc[p] == exc_approx(result.intra_exc[p]) );
      CHECK( r.VXCs[p].size() == 0 );
    }
    for( size_t j = 0; j < nprot; ++j )
      CHECK( r.inter_pair_exc[j] == exc_approx(result.inter_pair_exc[j]) );
  }

  // The terms-less overload fills every term in and must agree
  {
    auto r = integrator.eval_exc_vxc( densities, spec );
    for( size_t p = 0; p < nspecies; ++p ) {
      INFO( "species " << p );
      CHECK( r.intra_exc[p] == exc_approx(result.intra_exc[p]) );
      CHECK( (r.VXCs[p] - result.VXCs[p]).norm() / bases[p].nbf() < 1e-14 );
    }
    for( size_t j = 0; j < nprot; ++j )
      CHECK( r.inter_pair_exc[j] == exc_approx(result.inter_pair_exc[j]) );
  }

  // The answer must not depend on the order the protonic species are presented
  // in. The permutation reverses them and keeps the electron at index 0, so it
  // is its own inverse. Tolerances are one digit looser than above because the
  // per-task cost sort (and hence the reduction order) changes.
  if( nprot > 1 ) {
    std::vector<size_t> perm( nspecies );
    perm[0] = 0;
    for( size_t k = 1; k < nspecies; ++k ) perm[k] = nspecies - k;

    std::vector< BasisSet<double> > pbases;
    std::vector<mp_density_type>    pdens;
    MultiParticleFunctionalSpec     pspec;
    pspec.intra_functionals.resize( nspecies );
    for( size_t k = 0; k < nspecies; ++k ) {
      pbases.push_back( bases[perm[k]] );
      pdens.push_back( densities[perm[k]] );
      pspec.intra_functionals[k] = spec.intra_functionals[perm[k]];
    }
    for( const auto& pf : spec.inter_functionals )
      pspec.inter_functionals.push_back( MultiParticlePairFunctional{
        perm[pf.electron], perm[pf.particle], pf.functionals } );

    auto plb = lb_factory.get_instance(rt, mol, mg, pbases);
    mw.modify_weights(plb);
    auto pintegrator = integrator_factory.get_instance( func, plb );
    auto r = pintegrator.eval_exc_vxc( pdens, pspec, terms );

    for( size_t p = 0; p < nspecies; ++p ) {
      INFO( "species " << p << " -> " << perm[p] );
      CHECK( r.intra_exc[perm[p]] ==
             Approx(result.intra_exc[p]).epsilon(1e-12).margin(1e-14) );
      CHECK( (r.VXCs[perm[p]] - result.VXCs[p]).norm() / bases[p].nbf() < 1e-11 );
    }
    for( size_t j = 0; j < nprot; ++j ) {
      INFO( "EPC pair " << j );
      CHECK( r.inter_pair_exc[j] ==
             Approx(result.inter_pair_exc[j]).epsilon(1e-12).margin(1e-14) );
    }
  }

  // Check EXC Grad. The multiparticle gradient defaults to the full gradient
  // (weight derivatives included), matching the /EXC_GRAD convention above.
  if( check_grad and has_exc_grad ) {
    IntegratorSettingsEXC_GRAD exc_grad_settings;
    exc_grad_settings.include_weight_derivatives = true;
    auto EXC_GRAD = integrator.eval_exc_grad( densities, spec, terms,
      exc_grad_settings );
    using map_type = Eigen::Map<Eigen::MatrixXd>;
    map_type EXC_GRAD_ref_map( EXC_GRAD_ref.data(), mol.size(), 3 );
    map_type EXC_GRAD_map( EXC_GRAD.data(), mol.size(), 3 );
    auto EXC_GRAD_diff_nrm = (EXC_GRAD_ref_map - EXC_GRAD_map).norm();
    CHECK( EXC_GRAD_diff_nrm / std::sqrt(3.0*mol.size()) < 1e-8 );
  }

}

void test_neo_integrator(std::string reference_file, functional_type& func,
  functional_type& epc_func, PruningScheme pruning_scheme) {

#ifdef GAUXC_HAS_DEVICE
  auto rt = DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.9);
#else
  auto rt = RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
#endif

#ifdef GAUXC_HAS_HOST
    SECTION( "Host" ) {
      SECTION("Reference") {
        test_neo_xc_integrator( ExecutionSpace::Host, rt, reference_file, func,
          epc_func, pruning_scheme, true );
      }
    }
#endif

  // The device multiparticle path is not implemented; its contract is asserted
  // once, on a synthetic system, in "NEO XC Integrator / NYI + Validation".
}

/**
 *  A synthetic H2 + one protonic species system: small enough to build in
 *  memory, so the contract checks below need no reference file.
 */
struct NEOContractSystem {
  Molecule mol;
  std::vector< BasisSet<double> > bases;
  Eigen::MatrixXd P, Ps, Pz;

  NEOContractSystem() {
    mol = Molecule( std::vector<Atom>{
      Atom( AtomicNumber(1), 0.0, 0.0, -0.7 ),
      Atom( AtomicNumber(1), 0.0, 0.0,  0.7 ) } );

    auto s_shell = []( double alpha, std::array<double,3> O ) {
      Shell<double>::prim_array a{}, c{};
      a[0] = alpha; c[0] = 1.0;
      return Shell<double>( PrimSize(1), AngularMomentum(0), SphericalType(true),
        a, c, O, false );
    };

    // Electron: one 1s function per centre. Particle: one 1s function on the
    // first centre only.
    bases.push_back( BasisSet<double>( std::vector< Shell<double> >{
      s_shell( 1.0, {0.0, 0.0, -0.7} ), s_shell( 1.0, {0.0, 0.0, 0.7} ) } ) );
    bases.push_back( BasisSet<double>( std::vector< Shell<double> >{
      s_shell( 4.0, {0.0, 0.0, -0.7} ) } ) );

    P  = Eigen::MatrixXd::Identity(2,2);
    Ps = Eigen::MatrixXd::Identity(1,1);
    Pz = Eigen::MatrixXd::Identity(1,1);
  }

  std::vector<mp_density_type> densities() const {
    return { mp_density_type{ &P, nullptr }, mp_density_type{ &Ps, &Pz } };
  }
};


TEST_CASE( "XC Integrator", "[xc-integrator]" ) {

  auto pol     = ExchCXX::Spin::Polarized;
  auto unpol   = ExchCXX::Spin::Unpolarized;
  auto svwn5   = ExchCXX::Functional::SVWN5;
  auto pbe0    = ExchCXX::Functional::PBE0;
  auto blyp    = ExchCXX::Functional::BLYP;
  auto scan    = ExchCXX::Functional::SCAN;
  auto r2scanl = ExchCXX::Functional::R2SCANL;
  auto m062x   = ExchCXX::Functional::M062X;

  // LDA Test
  SECTION( "Benzene / SVWN5 / cc-pVDZ" ) {
    auto func = make_functional(svwn5, unpol);
    test_integrator(GAUXC_REF_DATA_PATH "/benzene_svwn5_cc-pvdz_ufg_ssf.hdf5", 
        func, PruningScheme::Unpruned );
  }
  SECTION( "Benzene / SVWN5 / cc-pVDZ (Treutler)" ) {
    auto func = make_functional(svwn5, unpol);
    test_integrator(GAUXC_REF_DATA_PATH "/benzene_svwn5_cc-pvdz_ufg_ssf_treutler_prune.hdf5", 
        func, PruningScheme::Treutler );
  }
  SECTION( "Benzene / SVWN5 / cc-pVDZ (Robust)" ) {
    auto func = make_functional(svwn5, unpol);
    test_integrator(GAUXC_REF_DATA_PATH "/benzene_svwn5_cc-pvdz_ufg_ssf_robust_prune.hdf5", 
        func, PruningScheme::Robust );
  }

  // GGA Test
  SECTION( "Benzene / PBE0 / cc-pVDZ" ) {
    auto func = make_functional(pbe0, unpol);
    test_integrator(GAUXC_REF_DATA_PATH "/benzene_pbe0_cc-pvdz_ufg_ssf.hdf5", 
        func, PruningScheme::Unpruned );
  }

  // MGGA Test (TAU Only)
  SECTION( "Cytosine / SCAN / cc-pVDZ") {
    auto func = make_functional(scan, unpol);
    test_integrator(GAUXC_REF_DATA_PATH "/cytosine_scan_cc-pvdz_ufg_ssf_robust.hdf5", 
        func, PruningScheme::Robust );
  }
  // This tests gradients
  SECTION( "Benzene / M06-2X / def2-svp") {
    auto func = make_functional(m062x, unpol);
    test_integrator(GAUXC_REF_DATA_PATH "/benzene_m062x_def2-svp_ufg_ssf.hdf5",
        func, PruningScheme::Unpruned );
  }

  // MGGA Test (TAU + LAPL)
  SECTION( "Cytosine / R2SCANL / cc-pVDZ") {
    auto func = make_functional(r2scanl, unpol);
    test_integrator(GAUXC_REF_DATA_PATH "/cytosine_r2scanl_cc-pvdz_ufg_ssf_robust.hdf5", 
        func, PruningScheme::Robust );
  }

  //UKS LDA Test
  SECTION( "Li / SVWN5 / sto-3g" ) {
    auto func = make_functional(svwn5, pol);
    test_integrator(GAUXC_REF_DATA_PATH "/li_svwn5_sto3g_uks.bin",
        func, PruningScheme::Unpruned );
  }
  // + grad
  SECTION( "Cytosine (doublet) / SVWN5 / cc-pVDZ") {
    auto func = make_functional(svwn5, pol);
    test_integrator(GAUXC_REF_DATA_PATH "/cytosine_svwn5_cc-pvdz_ufg_ssf_robust_uks.hdf5", 
        func, PruningScheme::Robust );
  }

  //UKS GGA Test
  SECTION( "Li / BLYP / sto-3g" ) {
    auto func = make_functional(blyp, pol);
    test_integrator(GAUXC_REF_DATA_PATH "/li_blyp_sto3g_uks.bin",
        func, PruningScheme::Unpruned );
  }
  // + grad
  SECTION( "Cytosine (doublet) / BLYP / cc-pVDZ") {
    auto func = make_functional(blyp, pol);
    test_integrator(GAUXC_REF_DATA_PATH "/cytosine_blyp_cc-pvdz_ufg_ssf_robust_uks.hdf5", 
        func, PruningScheme::Robust );
  }

  // UKS MGGA Test (TAU Only)
  SECTION( "Cytosine (doublet) / SCAN / cc-pVDZ") {
    auto func = make_functional(scan, pol);
    test_integrator(GAUXC_REF_DATA_PATH "/cytosine_scan_cc-pvdz_ufg_ssf_robust_uks.hdf5", 
        func, PruningScheme::Robust );
  }

  // UKS MGGA Test (TAU + LAPL)
  SECTION( "Cytosine (doublet) / R2SCANL / cc-pVDZ") {
    auto func = make_functional(r2scanl, pol);
    test_integrator(GAUXC_REF_DATA_PATH "/cytosine_r2scanl_cc-pvdz_ufg_ssf_robust_uks.hdf5", 
        func, PruningScheme::Robust );
  }

  // GKS GGA Test
  SECTION( "H3 / BLYP / cc-pvdz" ) {
    auto func = make_functional(blyp, pol);
    test_integrator(GAUXC_REF_DATA_PATH "/h3_blyp_cc-pvdz_ssf_gks.bin",
        func, PruningScheme::Unpruned );
  }

  // sn-LinK Test
  SECTION( "Benzene / PBE0 / 6-31G(d)" ) {
    auto func = make_functional(pbe0, unpol);
    test_integrator(GAUXC_REF_DATA_PATH "/benzene_631gd_pbe0_ufg.hdf5", 
        func, PruningScheme::Unpruned );
  }

  // sn-LinK + f functions
  SECTION( "H2O2 / PBE0 / def2-TZVP" ) {
    auto func = make_functional(pbe0, unpol);
    test_integrator(GAUXC_REF_DATA_PATH "/h2o2_def2-tzvp.hdf5", 
        func, PruningScheme::Unpruned );
  }

  // sn-LinK + g functions
  SECTION( "H2O2 / PBE0 / def2-QZVP" ) {
    auto func = make_functional(pbe0, unpol);
    test_integrator(GAUXC_REF_DATA_PATH "/h2o2_def2-qzvp.hdf5", 
        func, PruningScheme::Unpruned );
  }
}


TEST_CASE( "NEO XC Integrator", "[xc-integrator-neo]" ) {

  auto pol     = ExchCXX::Spin::Polarized;
  auto unpol   = ExchCXX::Spin::Unpolarized;
  auto blyp    = ExchCXX::Functional::BLYP;
  auto b3lyp   = ExchCXX::Functional::B3LYP;
  auto svwn5   = ExchCXX::Functional::SVWN5;
  auto epc17_2 = ExchCXX::Functional::EPC17_2;

  // One SECTION per reference file. The file carries the system, the densities
  // and the expected results; the intra/inter functionals and the pruning
  // scheme are hard-coded here -- exactly as the single-species sections above
  // do -- and must match what the reference was generated with.
  //
  // References are generated with standalone_driver in its NEO mode (see
  // tests/standalone_driver.cxx) from a converged ChronusQ NEO-SCF density,
  // using GRID=ULTRAFINE, RAD_QUAD=MURAKNOWLES, PRUNING_SCHEME=UNPRUNED,
  // BATCH_SIZE=512, BASIS_TOL=2.220446049250313e-16 and XC_WEIGHT_ALG=SSF,
  // i.e. exactly the settings test_neo_xc_integrator hard-codes above.

  // COH2 with one protonic species spanning both quantum hydrogens: a GGA
  // electronic intra functional against an LDA (EPC) inter functional. Carries
  // a gradient reference.
  SECTION( "COH2 / BLYP, EPC-17-2 / cc-pVDZ, prot-PB4-D" ) {
    auto func    = make_functional(blyp,    unpol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/coh2_blyp_epc17-2_cc-pvdz_pb4d_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

  // The same system with an LDA electronic functional. The LDA intra kernels
  // are a separate code path from the GGA ones above and are where the device
  // implementation is most likely to diverge. Carries a gradient reference.
  SECTION( "COH2 / SVWN5, EPC-17-2 / cc-pVDZ, prot-PB4-D" ) {
    auto func    = make_functional(svwn5,   unpol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/coh2_svwn5_epc17-2_cc-pvdz_pb4d_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

  // Smallest case (12 electronic / 8 protonic basis functions) and a different
  // protonic angular composition -- prot-SP is 1s1p where prot-PB4-D is
  // 4s3p2d -- which exercises L-dependent submatrix maps and screening.
  SECTION( "COH2 / BLYP, EPC-17-2 / STO-3G, prot-SP" ) {
    auto func    = make_functional(blyp,    unpol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/coh2_blyp_epc17-2_sto-3g_protsp_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

  // Distorted H2O with BOTH hydrogens quantum and DISTINGUISHABLE, i.e. two
  // protonic species and two EPC pairs sharing one electron. The O-H bonds are
  // 0.96 and 1.06 Angstrom at a 100 degree angle, so the two protonic species
  // carry genuinely different densities and different VXC (the reference has
  // |dP|_F/|P|_F = 0.53 between them): any species-index transposition changes
  // the answer. Two pairs also exercise EPC accumulation into the shared
  // electron channel, and nprot > 1 activates the permutation-invariance
  // block. Carries a gradient reference.
  SECTION( "H2O (distorted) / SVWN5, EPC-17-2 / cc-pVDZ, prot-PB4-D" ) {
    auto func    = make_functional(svwn5,   unpol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/h2o-distorted_svwn5_epc17-2_cc-pvdz_pb4d_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

  // A single quantum proton at one end of a near-linear molecule: the protonic
  // basis has support on one centre while the grid spans all three, so 55% of
  // the tasks carry no protonic basis functions at all. This is the screening /
  // empty-task case.
  SECTION( "HCN / BLYP, EPC-17-2 / cc-pVDZ, prot-PB4-D" ) {
    auto func    = make_functional(blyp,    unpol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/hcn_blyp_epc17-2_cc-pvdz_pb4d_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

  // Water dimer, one quantum proton per monomer as its own species. The two
  // protonic supports are ~3 Angstrom apart and disjoint -- 66% and 68% of the
  // tasks are empty for species 1 and 2 respectively -- which is the regime a
  // per-species active-task list has to get right.
  SECTION( "W02 water dimer / B3LYP, EPC-17-2 / cc-pVDZ, prot-PB4-D" ) {
    auto func    = make_functional(b3lyp,   unpol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/w02_b3lyp_epc17-2_cc-pvdz_pb4d_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

  // Water tetramer: 96 electronic basis functions, four protonic species and
  // four EPC pairs. The largest multi-pair scatter in the suite, and the only
  // case whose permutation-invariance block permutes more than two species.
  SECTION( "W04 water tetramer / B3LYP, EPC-17-2 / cc-pVDZ, prot-PB4-D" ) {
    auto func    = make_functional(b3lyp,   unpol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/w04_b3lyp_epc17-2_cc-pvdz_pb4d_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

  // The COH2 cation (charge +1, doublet) with an unrestricted electron. This is
  // the only NEO reference with a spin-polarised electron, and therefore the
  // only one that exercises the EPC potential being scattered into BOTH
  // electron spin channels; test_neo_xc_integrator picks it up from the
  // reference file's /DENSITY_Z, exactly as the single-species UKS sections do.
  SECTION( "COH2 cation (doublet) / BLYP, EPC-17-2 / cc-pVDZ, prot-PB4-D" ) {
    auto func    = make_functional(blyp,    pol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/coh2-cation_ublyp_epc17-2_cc-pvdz_pb4d_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

  // This section needs no reference data: it pins the multiparticle input
  // validation and the device NYI contract on a synthetic H2 + 1 protonic
  // species system.
  SECTION( "NYI + Validation" ) {

    using matrix_type = Eigen::MatrixXd;
#ifdef GAUXC_HAS_DEVICE
    auto rt = DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.9);
#else
    auto rt = RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
#endif

    NEOContractSystem sys;
    auto densities = sys.densities();

    auto mg = MolGridFactory::create_default_molgrid(sys.mol,
      PruningScheme::Unpruned, BatchSize(512), RadialQuad::MuraKnowles,
      AtomicGridSizeDefault::FineGrid);

    LoadBalancerFactory lb_factory( ExecutionSpace::Host, "Replicated" );
    auto lb = lb_factory.get_instance(rt, sys.mol, mg, sys.bases);
    REQUIRE( lb.basis_count() == 2 );

    MolecularWeightsFactory( ExecutionSpace::Host, "Default",
      MolecularWeightsSettings{} ).get_instance().modify_weights(lb);

    auto func     = make_functional(svwn5, unpol);
    auto epc_func = make_functional(epc17_2, pol);
    MultiParticleFunctionalSpec spec;
    spec.intra_functionals.resize(2);
    spec.intra_functionals[0].push_back( std::make_shared<functional_type>(func) );
    spec.inter_functionals.push_back( MultiParticlePairFunctional{ 0, 1,
      { std::make_shared<functional_type>(epc_func) } } );

    XCIntegratorFactory<matrix_type> integrator_factory( ExecutionSpace::Host,
      "Replicated", "Default", "Default", "Default" );
    auto integrator = integrator_factory.get_instance( func, lb );

    // The host path itself works on this system
    auto result = integrator.eval_exc_vxc( densities, spec );
    REQUIRE( result.intra_exc.size()      == 2 );
    REQUIRE( result.inter_pair_exc.size() == 1 );
    CHECK( result.VXCs[0].rows() == sys.bases[0].nbf() );
    CHECK( result.VXCs[1].rows() == sys.bases[1].nbf() );
    CHECK( result.inter_exc == Approx(result.inter_pair_exc[0]) );

    // One density per basis
    {
      std::vector<mp_density_type> bad = { densities[0] };
      CHECK_THROWS_WITH( integrator.eval_exc_vxc( bad, spec ),
        Catch::Contains("density count must match LoadBalancer basis count") );
    }

    // VXC targets are species indices
    {
      MultiParticleXCTerms terms;
      terms.active_intra = { 0, 1 };
      terms.active_inter = { 0 };
      terms.vxc_targets  = { 2 };
      CHECK_THROWS_WITH( integrator.eval_exc_vxc( densities, spec, terms ),
        Catch::Contains("Invalid MultiParticle VXC target index") );
    }

    // Inter-species GGA/mGGA (e.g. EPC19) is not implemented
    {
      auto gga_func = make_functional(blyp, pol);
      MultiParticleFunctionalSpec gga_spec = spec;
      gga_spec.inter_functionals[0].functionals =
        { std::make_shared<functional_type>(gga_func) };
      CHECK_THROWS_WITH( integrator.eval_exc_vxc( densities, gga_spec ),
        Catch::Contains("GGA/mGGA inter-XC is not implemented") );
    }

#ifdef GAUXC_HAS_DEVICE
    // The three seams a device multiparticle implementation has to remove.
    // When they are removed these checks fail, in the change that removes them.
    {
      LoadBalancerFactory dev_lb_factory( ExecutionSpace::Device, "Replicated" );
      CHECK_THROWS_WITH( dev_lb_factory.get_instance(rt, sys.mol, mg, sys.bases),
        Catch::Contains("does not support multiple basis sets") );

      XCIntegratorFactory<matrix_type> dev_factory( ExecutionSpace::Device,
        "Replicated", "Default", "Default", "Default" );
      auto dev_integrator = dev_factory.get_instance( func, lb );
      CHECK_THROWS_WITH( dev_integrator.eval_exc_vxc( densities, spec ),
        Catch::Contains("MultiParticle EXC/VXC is not implemented") );
      CHECK_THROWS_WITH( dev_integrator.eval_exc_grad( densities, spec ),
        Catch::Contains("MultiParticle EXC Gradient is not implemented") );
    }
#endif
  }
}

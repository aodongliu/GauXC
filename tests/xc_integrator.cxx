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


// A NEO reference file carries the single-species datasets for the electron,
// which is species 0, and per protonic species i (GauXC species i+1):
//   /PROTONIC_BASIS_<i>            /PROTONIC_VXC_SCALAR_<i>
//   /PROTONIC_DENSITY_SCALAR_<i>   /PROTONIC_VXC_Z_<i>
//   /PROTONIC_DENSITY_Z_<i>        /PROTONIC_EXC_<i>
//   /EPC_EXC_<j>                   pair j = (electron, species j+1)

using multiparticle_density_type =
  XCIntegrator<Eigen::MatrixXd>::multiparticle_density;

std::string neo_dataset( std::string base, size_t i ) {
  return base + std::to_string(i);
}


// The system a NEO reference file describes: molecule, bases and densities.
struct neo_reference {
  Molecule                        mol;
  std::vector< BasisSet<double> > bases;
  Eigen::MatrixXd                 Ps, Pz;
  std::vector< Eigen::MatrixXd >  prot_Ps, prot_Pz;
  bool                            uks = false;
  std::vector<bool>               prot_uks;

  size_t nspecies() const { return bases.size();     }
  size_t nprot()    const { return bases.size() - 1; }

  std::vector<multiparticle_density_type> densities() const {
    std::vector<multiparticle_density_type> d;
    d.push_back( { &Ps, uks ? &Pz : nullptr } );
    for( size_t i = 0; i < nprot(); ++i )
      d.push_back( { &prot_Ps[i], prot_uks[i] ? &prot_Pz[i] : nullptr } );
    return d;
  }
};

neo_reference read_neo_reference( std::string reference_file ) {

  neo_reference ref;
  size_t nprot = 0;
  {
    HighFive::File file( reference_file, HighFive::File::ReadOnly );
    ref.uks = file.exist("/DENSITY_Z");
    while( file.exist( neo_dataset("/PROTONIC_BASIS_", nprot) ) ) nprot++;
  }
  REQUIRE( nprot > 0 );

  read_hdf5_record( ref.mol, reference_file, "/MOLECULE" );
  ref.bases.resize( nprot + 1 );
  read_hdf5_record( ref.bases[0], reference_file, "/BASIS" );
  for( size_t i = 0; i < nprot; ++i )
    read_hdf5_record( ref.bases[i+1], reference_file,
      neo_dataset("/PROTONIC_BASIS_", i) );
  for( auto& b : ref.bases ) for( auto& sh : b )
    sh.set_shell_tolerance( std::numeric_limits<double>::epsilon() );

  HighFive::File file( reference_file, HighFive::File::ReadOnly );
  auto read_matrix = [&]( std::string name ) {
    auto dset = file.getDataSet( name );
    auto dims = dset.getDimensions();
    Eigen::MatrixXd M( dims[0], dims[1] );
    dset.read( M.data() );
    return M;
  };

  ref.Ps = read_matrix( ref.uks ? "/DENSITY_SCALAR" : "/DENSITY" );
  if( ref.uks ) ref.Pz = read_matrix( "/DENSITY_Z" );

  ref.prot_Ps.resize( nprot );
  ref.prot_Pz.resize( nprot );
  ref.prot_uks.assign( nprot, false );
  for( size_t i = 0; i < nprot; ++i ) {
    ref.prot_Ps[i]  = read_matrix( neo_dataset("/PROTONIC_DENSITY_SCALAR_", i) );
    ref.prot_uks[i] = file.exist( neo_dataset("/PROTONIC_DENSITY_Z_", i) );
    if( ref.prot_uks[i] )
      ref.prot_Pz[i] = read_matrix( neo_dataset("/PROTONIC_DENSITY_Z_", i) );
  }

  return ref;
}

void test_neo_xc_integrator( ExecutionSpace ex, const RuntimeEnvironment& rt,
  std::string reference_file,
  functional_type& func,      // the electron's intra functional
  functional_type& epc_func,  // the electron/particle pair functional
  PruningScheme pruning_scheme,
  bool check_grad,
  std::string lwd_kernel = "Default" ) {

  using matrix_type = Eigen::MatrixXd;

  auto ref = read_neo_reference( reference_file );
  const size_t nprot    = ref.nprot();
  const size_t nspecies = ref.nspecies();
  const auto&  mol      = ref.mol;
  const auto&  bases    = ref.bases;

  matrix_type VXC_ref, VXCz_ref;
  std::vector<matrix_type> prot_VXCs_ref(nprot), prot_VXCz_ref(nprot);
  std::vector<double> prot_EXC_ref(nprot), EPC_EXC_ref(nprot);
  std::vector<double> EXC_GRAD_ref_HellFey(3*mol.size()), EXC_GRAD_ref_Full(3*mol.size());
  double EXC_ref = 0.;

  {
    HighFive::File file( reference_file, HighFive::File::ReadOnly );

    auto read_matrix = [&]( std::string name ) {
      auto dset = file.getDataSet( name );
      auto dims = dset.getDimensions();
      matrix_type M( dims[0], dims[1] );
      dset.read( M.data() );
      return M;
    };

    VXC_ref = read_matrix( ref.uks ? "/VXC_SCALAR" : "/VXC" );
    if( ref.uks ) VXCz_ref = read_matrix("/VXC_Z");
    file.getDataSet("/EXC").read( &EXC_ref );

    for( size_t i = 0; i < nprot; ++i ) {
      prot_VXCs_ref[i] = read_matrix( neo_dataset("/PROTONIC_VXC_SCALAR_", i) );
      if( ref.prot_uks[i] )
        prot_VXCz_ref[i] = read_matrix( neo_dataset("/PROTONIC_VXC_Z_", i) );
      file.getDataSet( neo_dataset("/PROTONIC_EXC_", i) ).read( &prot_EXC_ref[i] );
      file.getDataSet( neo_dataset("/EPC_EXC_", i)      ).read( &EPC_EXC_ref[i]  );
    }

    file.getDataSet("/EXC_GRAD").read( EXC_GRAD_ref_Full.data() );
    file.getDataSet("/EXC_GRAD_HELLFEY").read( EXC_GRAD_ref_HellFey.data() );
  }

  auto mg = MolGridFactory::create_default_molgrid(mol, pruning_scheme,
    BatchSize(512), RadialQuad::MuraKnowles, AtomicGridSizeDefault::UltraFineGrid);

  LoadBalancerFactory lb_factory(ExecutionSpace::Host, "Default");
  auto lb = lb_factory.get_instance(rt, mol, mg, bases);

  MolecularWeightsFactory mw_factory( ex, "Default", MolecularWeightsSettings{} );
  auto mw = mw_factory.get_instance();
  mw.modify_weights(lb);

  // An intra functional on the electron, one pair functional per species
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

  XCIntegratorFactory<matrix_type> integrator_factory( ex, "Replicated",
    "Default", lwd_kernel, "Default" );
  auto integrator = integrator_factory.get_instance( func, lb );

  auto densities = ref.densities();

  auto exc_approx = []( double x ) { return Approx(x).epsilon(1e-10).margin(1e-14); };

  auto result = integrator.eval_exc_vxc( densities, spec, terms );

  // Shape
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

  // Per-pair EPC EXC, each pair and their sum
  double epc_sum = 0.;
  for( size_t j = 0; j < nprot; ++j ) {
    INFO( "EPC pair " << j );
    CHECK( result.inter_pair_exc[j] == exc_approx(EPC_EXC_ref[j]) );
    epc_sum += result.inter_pair_exc[j];
  }
  CHECK( result.inter_exc == exc_approx(epc_sum) );

  // Per-species, per-channel VXC
  CHECK( (result.VXCs[0] - VXC_ref).norm() / bases[0].nbf() < 1e-10 );
  if( ref.uks ) CHECK( (result.VXCz[0] - VXCz_ref).norm() / bases[0].nbf() < 1e-10 );
  for( size_t i = 0; i < nprot; ++i ) {
    INFO( "protonic species " << i );
    const auto pnbf = bases[i+1].nbf();
    CHECK( (result.VXCs[i+1] - prot_VXCs_ref[i]).norm() / pnbf < 1e-10 );
    if( ref.prot_uks[i] )
      CHECK( (result.VXCz[i+1] - prot_VXCz_ref[i]).norm() / pnbf < 1e-10 );
  }

  // The EXC gradient, weight derivatives included
  std::vector<double> EXC_GRAD_full;
  auto neo_exc_grad = [&]( auto& itg, bool weight_derivatives ) {
    IntegratorSettingsEXC_GRAD s;
    s.include_weight_derivatives = weight_derivatives;
    return itg.eval_exc_grad( densities, spec, terms, s );
  };

  if( check_grad ) {
    EXC_GRAD_full = neo_exc_grad( integrator, true );
    REQUIRE( EXC_GRAD_full.size() == 3*mol.size() );

    using map_type = Eigen::Map<Eigen::MatrixXd>;
    auto check_exc_grad = [&]( std::vector<double>& EXC_GRAD, std::vector<double>& EXC_GRAD_ref ) {
      map_type EXC_GRAD_ref_map( EXC_GRAD_ref.data(), mol.size(), 3 );
      map_type EXC_GRAD_map( EXC_GRAD.data(), mol.size(), 3 );
      auto EXC_GRAD_diff_nrm = (EXC_GRAD_ref_map - EXC_GRAD_map).norm();
      CHECK( EXC_GRAD_diff_nrm / std::sqrt(3.0*mol.size()) < 1e-8 );
    };
    check_exc_grad( EXC_GRAD_full, EXC_GRAD_ref_Full );

    // Without weight derivatives
    auto EXC_GRAD_HellFey = neo_exc_grad( integrator, false );
    check_exc_grad( EXC_GRAD_HellFey, EXC_GRAD_ref_HellFey );
  }

  // The answer does not depend on the order the protonic species are given in
  if( nprot > 1 ) {
    std::vector<size_t> perm( nspecies );
    perm[0] = 0;
    for( size_t k = 1; k < nspecies; ++k ) perm[k] = nspecies - k;

    std::vector< BasisSet<double> >          pbases;
    std::vector<multiparticle_density_type>  pdens;
    MultiParticleFunctionalSpec              pspec;
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

    if( check_grad ) {
      IntegratorSettingsEXC_GRAD pgs;
      pgs.include_weight_derivatives = true;
      auto pg = pintegrator.eval_exc_grad( pdens, pspec, terms, pgs );
      REQUIRE( pg.size() == EXC_GRAD_full.size() );
      double worst = 0.;
      for( size_t k = 0; k < pg.size(); ++k )
        worst = std::max( worst, std::abs( pg[k] - EXC_GRAD_full[k] ) );
      INFO( "worst per-atom deviation under species permutation " << worst );
      CHECK( worst < 1e-8 );
    }
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
    test_neo_xc_integrator( ExecutionSpace::Host, rt, reference_file, func,
      epc_func, pruning_scheme, true );
  }
#endif

#ifdef GAUXC_HAS_DEVICE
  // Only the Incore integrator runs the multiparticle path: ShellBatched and
  // the MAGMA local work driver refuse it with an explicit NYI throw
  SECTION( "Device" ) {
    SECTION( "Incore - MPI Reduction" ) {
      test_neo_xc_integrator( ExecutionSpace::Device, rt, reference_file, func,
        epc_func, pruning_scheme, true );
    }

    #ifdef GAUXC_HAS_CUTLASS
    SECTION( "Incore - MPI Reduction - CUTLASS" ) {
      test_neo_xc_integrator( ExecutionSpace::Device, rt, reference_file, func,
        epc_func, pruning_scheme, true, "Scheme1-CUTLASS" );
    }
    #endif
  }
#endif
}


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
  auto pbe     = ExchCXX::Functional::PBE;
  auto pbe0    = ExchCXX::Functional::PBE0;
  auto svwn5   = ExchCXX::Functional::SVWN5;
  auto epc17_2 = ExchCXX::Functional::EPC17_2;

  // Reference settings: ULTRAFINE, MuraKnowles, unpruned, batch 512, SSF

  SECTION( "HCN / BLYP, EPC-17-2 / cc-pVDZ, prot-PB4-D" ) {
    auto func    = make_functional(blyp,    unpol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/hcn_blyp_epc17-2_cc-pvdz_pb4d_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

  SECTION( "COH2 / SVWN5, EPC-17-2 / STO-3G, prot-SP, one protonic species" ) {
    auto func    = make_functional(svwn5,   unpol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/coh2_svwn5_epc17-2_sto-3g_protsp_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

  SECTION( "COH2 / SVWN5, EPC-17-2 / STO-3G, prot-SP, two protonic species" ) {
    auto func    = make_functional(svwn5,   unpol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/coh2_svwn5_epc17-2_sto-3g_protsp-dist_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

  SECTION( "COH2 cation (doublet) / PBE, EPC-17-2 / cc-pVDZ, prot-PB4-D" ) {
    auto func    = make_functional(pbe,     pol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/coh2-cation_upbe_epc17-2_cc-pvdz_pb4d_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

  SECTION( "W08 water octamer, 16 quantum protons / B3LYP, EPC-17-2 / cc-pVDZ, prot-SP" ) {
    auto func    = make_functional(b3lyp,   unpol);
    auto epcfunc = make_functional(epc17_2, pol);
    test_neo_integrator(GAUXC_REF_DATA_PATH
      "/w08_b3lyp_epc17-2_cc-pvdz_16protsp_ssf.hdf5", func, epcfunc,
      PruningScheme::Unpruned );
  }

#ifdef GAUXC_HAS_DEVICE
  // One species and no pairs: the multiparticle entry point equals the plain one
  SECTION( "one species reduces to the single-species path" ) {

    using matrix_type = Eigen::MatrixXd;
    auto rt = DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.9);

    auto reduces = [&]( std::string reference_file, functional_type& func ) {
      Molecule mol;
      BasisSet<double> basis;
      read_hdf5_record( mol,   reference_file, "/MOLECULE" );
      read_hdf5_record( basis, reference_file, "/BASIS"    );
      for( auto& sh : basis )
        sh.set_shell_tolerance( std::numeric_limits<double>::epsilon() );

      matrix_type P;
      {
        HighFive::File file( reference_file, HighFive::File::ReadOnly );
        auto dset = file.getDataSet("/DENSITY");
        auto dims = dset.getDimensions();
        P = matrix_type( dims[0], dims[1] );
        dset.read( P.data() );
      }

      auto mg = MolGridFactory::create_default_molgrid(mol,
        PruningScheme::Unpruned, BatchSize(512), RadialQuad::MuraKnowles,
        AtomicGridSizeDefault::UltraFineGrid);
      LoadBalancerFactory lb_factory( ExecutionSpace::Host, "Default" );
      auto lb = lb_factory.get_instance( rt, mol, mg, basis );
      MolecularWeightsFactory( ExecutionSpace::Device, "Default",
        MolecularWeightsSettings{} ).get_instance().modify_weights(lb);

      XCIntegratorFactory<matrix_type> factory( ExecutionSpace::Device,
        "Replicated", "Default", "Default", "Default" );
      auto integrator = factory.get_instance( func, lb );

      double EXC = 0.;
      matrix_type VXC;
      std::tie( EXC, VXC ) = integrator.eval_exc_vxc( P );

      MultiParticleFunctionalSpec spec;
      spec.intra_functionals.resize(1);
      spec.intra_functionals[0].push_back( std::make_shared<functional_type>(func) );
      std::vector<multiparticle_density_type> densities = {
        multiparticle_density_type{ &P, nullptr } };
      auto r = integrator.eval_exc_vxc( densities, spec );

      REQUIRE( r.intra_exc.size() == 1 );
      CHECK( r.inter_exc == 0. );
      CHECK( r.intra_exc[0] == Approx(EXC).epsilon(1e-10).margin(1e-14) );
      CHECK( (r.VXCs[0] - VXC).norm() / basis.nbf() < 1e-10 );
    };

    SECTION( "Benzene / SVWN5 / cc-pVDZ" ) {
      auto func = make_functional(svwn5, unpol);
      reduces( GAUXC_REF_DATA_PATH "/benzene_svwn5_cc-pvdz_ufg_ssf.hdf5", func );
    }

    SECTION( "Benzene / PBE0 / cc-pVDZ" ) {
      auto func = make_functional(pbe0, unpol);
      reduces( GAUXC_REF_DATA_PATH "/benzene_pbe0_cc-pvdz_ufg_ssf.hdf5", func );
    }
  }
#endif
}

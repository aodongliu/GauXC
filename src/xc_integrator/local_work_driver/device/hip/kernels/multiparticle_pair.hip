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
#include "device/common/multiparticle_pair.hpp"
#include <gauxc/exceptions.hpp>

namespace GauXC {

/*
 * Placeholders until the multiparticle pair kernels are ported to HIP. The
 * backend-generic AoSScheme1Base calls every entry point below, so a HIP build
 * has to resolve them; each one throws.
 */

void gather_pair_densities( size_t, const pair_point_range*, const double*,
  const double*, double*, device_queue ) {
  GAUXC_GENERIC_EXCEPTION("MultiParticle pair XC is NYI for HIP");
}

void scatter_pair_potential( size_t, const pair_point_range*, const double*,
  const double*, double*, double*, device_queue ) {
  GAUXC_GENERIC_EXCEPTION("MultiParticle pair XC is NYI for HIP");
}

void sum_pair_energy( size_t, const pair_point_range*, const double*,
  const double*, const double*, double*, device_queue ) {
  GAUXC_GENERIC_EXCEPTION("MultiParticle pair EXC is NYI for HIP");
}

void add_intra_w_times_f( size_t, const XCDeviceTask*, const size_t*, double*,
  device_queue ) {
  GAUXC_GENERIC_EXCEPTION("MultiParticle EXC Gradient is NYI for HIP");
}

void add_pair_w_times_f( size_t, const pair_point_range*, const double*,
  const double*, const double*, double*, device_queue ) {
  GAUXC_GENERIC_EXCEPTION("MultiParticle EXC Gradient is NYI for HIP");
}

}

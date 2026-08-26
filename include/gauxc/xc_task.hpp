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

#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <gauxc/gauxc_config.hpp>
#include <gauxc/shell.hpp>
#include <gauxc/exceptions.hpp>

namespace GauXC {

struct XCTask {

  int32_t                              iParent = -1;
  std::vector< std::array<double,3> >  points;
  std::vector< double  >               weights;
  int32_t                              npts = 0;

  double                               dist_nearest;
  double                               max_weight = std::numeric_limits<double>::infinity();

  struct screening_data {
    using pair_t = std::pair<int32_t,int32_t>;
    std::vector<int32_t>               shell_list;
    std::vector<pair_t>                shell_pair_list;
    std::vector<int32_t>               shell_pair_idx_list;
    std::vector<int32_t>               submat_block;
    std::vector<std::array<int32_t,3>> submat_map;
    int32_t                            nbe = 0;

    bool equiv_with( const screening_data& other ) const {
      return shell_list == other.shell_list and 
        shell_pair_list == other.shell_pair_list;
    }

    inline size_t volume() const {
      return (shell_list.size() + 2*shell_pair_list.size() + submat_block.size() +
              3*submat_map.size() + 1) * sizeof(int32_t);
    }
  };

  inline size_t volume() const {
    const auto bfn_volume = bfn_screenings.empty() ?
      bfn_screening.volume() :
      std::accumulate( bfn_screenings.begin(), bfn_screenings.end(), size_t{0},
        []( const auto& v, const auto& s ) { return v + s.volume(); } );

    return 2 * sizeof(int32_t) +
      (3*points.size() + weights.size() + 2) * sizeof(double) +
      bfn_volume + cou_screening.volume();
  }

  screening_data bfn_screening;
  std::vector<screening_data> bfn_screenings;
  screening_data cou_screening;

  const screening_data& basis_screening( size_t i ) const {
    if( bfn_screenings.empty() ) {
      if( i == 0 ) return bfn_screening;
      GAUXC_GENERIC_EXCEPTION("Requested basis screening is not available");
    }
    return bfn_screenings.at(i);
  }

  screening_data& basis_screening( size_t i ) {
    if( bfn_screenings.empty() )
      bfn_screenings.push_back( bfn_screening );
    return bfn_screenings.at(i);
  }

  void merge_with( const XCTask& other ) {
    if( !equiv_with(other) )
      GAUXC_GENERIC_EXCEPTION("Cannot Perform Requested Merge: Incompatible Tasks");
    points.insert( points.end(), other.points.begin(), other.points.end() );
    weights.insert( weights.end(), other.weights.begin(), other.weights.end() );
    npts = static_cast<int32_t>(points.size());
  }

  template <typename TaskIt>
  void merge_with( TaskIt begin, TaskIt end ) {

    size_t old_sz = points.size();
    size_t pts_add = std::accumulate( begin, end, size_t{0},
      []( const auto &a, const auto &t ) {
        return a + t.points.size();
      });

    size_t new_sz = old_sz + pts_add;
    points.resize( new_sz );
    weights.resize( new_sz );

    auto points_it  = points.begin()  + old_sz;
    auto weights_it = weights.begin() + old_sz;
    for( auto it = begin; it != end; ++it ) {
      if( !equiv_with(*it) )
        GAUXC_GENERIC_EXCEPTION("Cannot Perform Requested Task Merge");
      points_it  = std::copy( it->points.begin(), it->points.end(), points_it );
      weights_it = std::copy( it->weights.begin(), it->weights.end(), weights_it );
    }

    npts = static_cast<int32_t>(points.size());
  }

  inline bool equiv_with( const XCTask& other ) const {
    if( iParent != other.iParent ) return false;

    if( bfn_screenings.empty() and other.bfn_screenings.empty() )
      return bfn_screening.equiv_with(other.bfn_screening);

    if( bfn_screenings.size() != other.bfn_screenings.size() )
      return false;

    return std::equal( bfn_screenings.begin(), bfn_screenings.end(),
      other.bfn_screenings.begin(),
      []( const auto& a, const auto& b ) { return a.equiv_with(b); } );
  }

  template <typename Archive>
  void serialize( Archive& ar ) {
    ar( iParent, bfn_screening.nbe, npts, dist_nearest, max_weight, 
      bfn_screening.shell_list, points, weights );  
  }


  // Per-point basis work estimate, summed over the active basis sets:
  //   sum_p nbe_p * ( 1 + nbe_p + n_deriv )
  // The collocation/X-matrix/Z-matrix work of each species is quadratic in
  // that species' own nbe; accumulating nbe over species and squaring the
  // total would add a spurious cross term 2*nbe_e*nbe_p that corresponds to
  // no actual work. For a single species (or the legacy single-screening
  // path) this is algebraically identical to the previous expression.
  inline size_t bfn_work(size_t n_deriv) const {
    if( bfn_screenings.empty() ) {
      const size_t nbe = bfn_screening.nbe;
      return nbe * ( 1 + nbe + n_deriv );
    }
    size_t w = 0;
    for( const auto& s : bfn_screenings ) {
      const size_t nbe = s.nbe;
      w += nbe * ( 1 + nbe + n_deriv );
    }
    return w;
  }

  inline size_t cost(size_t n_deriv, size_t natoms) const {
    return (bfn_work(n_deriv) + natoms * natoms) * npts;
  }
  inline size_t cost_exc_vxc(size_t n_deriv) const {
    return bfn_work(n_deriv) * npts;
  }
  inline size_t cost_exx() const {
    return ( bfn_screening.nbe + 2*cou_screening.nbe*bfn_screening.nbe +
             2*cou_screening.shell_pair_list.size() ) * npts;
  }
};


}

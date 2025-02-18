#pragma once
#include "framework/mesh/mesh.h"

namespace pdes
{
  /**
   * Creates a 1D orthogonal mesh in the given coordinate system with the
   * given 1D vertices.
   */
  Mesh create_1d_orthomesh(const std::vector<double>& x_coords,
                           Mesh::CoordinateSystem coord_sys =
                               Mesh::CoordinateSystem::CARTESIAN);

  /**
   * Creates a 1D orthogonal mesh in the given coordinate system subdivided
   * into regions with a varying number of cells per region.
   */
  Mesh create_1d_orthomesh(const std::vector<double>& region_edges,
                           const std::vector<unsigned int>& cells_per_regions,
                           const std::vector<unsigned int>& block_ids,
                           Mesh::CoordinateSystem coord_sys =
                               Mesh::CoordinateSystem::CARTESIAN);

  /**
   * Creates a 2D orthogonal mesh in the given coordinate system with the
   * outer product of the give x and y-coordinates.
   */
  Mesh create_2d_orthomesh(const std::vector<double>& x_coords,
                           const std::vector<double>& y_coords,
                           Mesh::CoordinateSystem coord_sys =
                               Mesh::CoordinateSystem::CARTESIAN);

  /**
   * Creates a 2D orthogonal mesh in the given coordinate system subdivided
   * into regions in each dimension with a varying number of cells per region
   * per dimension.
   */
  Mesh create_2d_orthomesh(const std::vector<double>& x_region_edges,
                           const std::vector<double>& y_region_edges,
                           const std::vector<unsigned int>& cells_per_x_region,
                           const std::vector<unsigned int>& cells_per_y_region,
                           const std::vector<unsigned int>& block_ids,
                           Mesh::CoordinateSystem coord_sys =
                               Mesh::CoordinateSystem::CARTESIAN);

  /**
   * Creates a 3D orthogonal mesh in the given coordinate system with the
   * outer product of the give x, y, and z-coordinates.
   */
  Mesh create_3d_orthomesh(const std::vector<double>& x_coords,
                           const std::vector<double>& y_coords,
                           const std::vector<double>& z_coords,
                           Mesh::CoordinateSystem coord_sys =
                               Mesh::CoordinateSystem::CARTESIAN);

  /**
   * Creates a 3D orthogonal mesh in the given coordinate system subdivided
   * into regions in each dimension with a varying number of cells per region
   * per dimension.
   */
  Mesh create_3d_orthomesh(const std::vector<double>& x_region_edges,
                           const std::vector<double>& y_region_edges,
                           const std::vector<double>& z_region_edges,
                           const std::vector<unsigned int>& cells_per_x_region,
                           const std::vector<unsigned int>& cells_per_y_region,
                           const std::vector<unsigned int>& cells_per_z_region,
                           const std::vector<unsigned int>& block_ids,
                           Mesh::CoordinateSystem coord_sys =
                               Mesh::CoordinateSystem::CARTESIAN);
}

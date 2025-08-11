#pragma once
#include "framework/mesh/mesh.h"

namespace pdes
{
  class OrthoMeshGenerator
  {
  public:
    /**
     * Creates a 1D orthogonal mesh in the given coordinate system with the
     * given 1D vertices.
     */
    static Mesh create_1d_orthomesh(const std::vector<types::real>& x_coords,
                                    Mesh::CoordinateSystem coord_sys =
                                        Mesh::CoordinateSystem::CARTESIAN);

    /**
     * Creates a 1D orthogonal mesh in the given coordinate system subdivided
     * into regions with a varying number of cells per region.
     */
    static Mesh create_1d_orthomesh(const std::vector<types::real>& region_edges,
                                    const std::vector<unsigned int>& cells_per_regions,
                                    const std::vector<unsigned int>& block_ids,
                                    Mesh::CoordinateSystem coord_sys =
                                        Mesh::CoordinateSystem::CARTESIAN);

    /**
     * Creates a 2D orthogonal mesh in the given coordinate system with the
     * outer product of the given x and y coordinates.
     */
    static Mesh create_2d_orthomesh(const std::vector<types::real>& x_coords,
                                    const std::vector<types::real>& y_coords,
                                    Mesh::CoordinateSystem coord_sys =
                                        Mesh::CoordinateSystem::CARTESIAN);

    /**
     * Creates a 2D orthogonal mesh in the given coordinate system subdivided
     * into regions in each dimension with a varying number of cells per region
     * per dimension.
     */
    static Mesh create_2d_orthomesh(const std::vector<types::real>& x_region_edges,
                                    const std::vector<types::real>& y_region_edges,
                                    const std::vector<unsigned int>& cells_per_x_region,
                                    const std::vector<unsigned int>& cells_per_y_region,
                                    const std::vector<unsigned int>& block_ids,
                                    Mesh::CoordinateSystem coord_sys =
                                        Mesh::CoordinateSystem::CARTESIAN);

    /**
     * Creates a 3D orthogonal mesh in the given coordinate system with the
     * outer product of the given x, y, and z coordinates.
     */
    static Mesh create_3d_orthomesh(const std::vector<types::real>& x_coords,
                                    const std::vector<types::real>& y_coords,
                                    const std::vector<types::real>& z_coords,
                                    Mesh::CoordinateSystem coord_sys =
                                        Mesh::CoordinateSystem::CARTESIAN);

    /**
     * Creates a 3D orthogonal mesh in the given coordinate system subdivided
     * into regions in each dimension with a varying number of cells per region
     * per dimension.
     */
    static Mesh create_3d_orthomesh(const std::vector<types::real>& x_region_edges,
                                    const std::vector<types::real>& y_region_edges,
                                    const std::vector<types::real>& z_region_edges,
                                    const std::vector<unsigned int>& cells_per_x_region,
                                    const std::vector<unsigned int>& cells_per_y_region,
                                    const std::vector<unsigned int>& cells_per_z_region,
                                    const std::vector<unsigned int>& block_ids,
                                    Mesh::CoordinateSystem coord_sys =
                                        Mesh::CoordinateSystem::CARTESIAN);
  };
}

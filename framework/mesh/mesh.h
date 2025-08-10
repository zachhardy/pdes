#pragma once
#include "framework/types.h"
#include "framework/mesh/cell.h"
#include "framework/mesh/mesh_vector.h"
#include <map>
#include <vector>

namespace pdes
{
  class Mesh
  {
  public:
    using VertexMap = std::map<types::global_index, MeshVector<>>;

    enum class CoordinateSystem
    {
      CARTESIAN = 0, ///< @f$ x, y, z @f$ coordinates
      CYLINDRICAL = 1, ///< @f$ r, z, \theta @f$ coordinates
      SPHERICAL = 2 /// @f$ \rho, \theta, \varphi @f$ coordinates
    };

    struct OrthoAttributes
    {
      unsigned int nx = 0; ///< Number of cells in the x-dimension
      unsigned int ny = 0; ///< Number of cells in the y-dimension
      unsigned int nz = 0; ///< Number of cells in the z-dimension
    };

    Mesh(const unsigned int dim, const CoordinateSystem coord_sys)
      : dim_(dim),
        coord_sys_(coord_sys),
        orthogonal_(false),
        extruded_(false)
    {}

    unsigned int dimension() const { return dim_; }
    CoordinateSystem coordinate_system() const { return coord_sys_; }

    void set_ortho_attributes(unsigned int nx,
                              unsigned int ny = 0,
                              unsigned int nz = 0);
    OrthoAttributes ortho_attributes() const { return ortho_attr_; }

    bool is_orthogonal() const { return orthogonal_; }
    bool is_extruded() const { return extruded_; }

    void add_vertex(types::global_index i, MeshVector<> vertex, bool allow_replace = false);
    void add_vertices(VertexMap vertices, bool allow_replace = false);

    void add_cell(Cell cell) { local_cells_.push_back(std::move(cell)); }
    void add_cells(std::vector<Cell> cells);

    unsigned int num_vertices() const { return vertices_.size(); }
    const MeshVector<>& vertex(const types::global_index i) const { return vertices_.at(i); }
    const std::map<types::global_index, MeshVector<>>& vertices() const { return vertices_; }

    unsigned int num_local_cells() const { return local_cells_.size(); }
    Cell& local_cell(const unsigned int i) { return local_cells_.at(i); }
    const Cell& local_cell(const unsigned int local_id) const { return local_cells_.at(local_id); }
    std::vector<Cell>& local_cells() { return local_cells_; }
    const std::vector<Cell>& local_cells() const { return local_cells_; }

  private:
    const unsigned int dim_;
    const CoordinateSystem coord_sys_;

    bool orthogonal_;
    OrthoAttributes ortho_attr_;

    bool extruded_;

    std::vector<Cell> local_cells_;
    VertexMap vertices_;
  };
}

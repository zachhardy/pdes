#pragma once
#include "framework/mesh/cell.h"
#include "framework/mesh/mesh_vector.h"
#include <vector>

namespace pdes
{
  class Mesh
  {
  public:
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

    Mesh(unsigned int dim, CoordinateSystem coord_sys);

    unsigned int dimension() const { return dim_; }
    CoordinateSystem coordinate_system() const { return coord_sys_; }

    void set_ortho_attributes(unsigned int nx,
                              unsigned int ny = 0,
                              unsigned int nz = 0);
    OrthoAttributes ortho_attributes() const { return ortho_attr_; }

    bool is_orthogonal() const { return orthogonal_; }
    bool is_extruded() const { return extruded_; }

    void add_vertex(MeshVector<>&& vertex) { vertices_.push_back(vertex); }
    void add_vertices(std::vector<MeshVector<>>&& vertices);

    MeshVector<>& vertices(const unsigned int i) { return vertices_.at(i); }
    const MeshVector<>& vertices(unsigned int i) const;

    std::vector<MeshVector<>>& vertices() { return vertices_; }
    const std::vector<MeshVector<>>& vertices() const { return vertices_; }

    void add_cell(Cell&& cell) { cells_.push_back(cell); }
    void add_cells(std::vector<Cell>&& cells);

    Cell& cells(const unsigned int local_id) { return cells_.at(local_id); }
    const Cell& cells(unsigned int local_id) const;

    std::vector<Cell>& cells() { return cells_; }
    const std::vector<Cell>& cells() const { return cells_; }

  private:
    const unsigned int dim_;
    const CoordinateSystem coord_sys_;

    bool orthogonal_ = false;
    OrthoAttributes ortho_attr_;

    bool extruded_ = false;

    std::vector<MeshVector<>> vertices_;
    std::vector<Cell> cells_;
  };

  inline void
  Mesh::set_ortho_attributes(const unsigned int nx,
                             const unsigned int ny,
                             const unsigned int nz)
  {
    orthogonal_ = true;
    ortho_attr_.nx = nx;
    ortho_attr_.ny = ny;
    ortho_attr_.nz = nz;
  }

  inline const MeshVector<>&
  Mesh::vertices(const unsigned int i) const
  {
    return vertices_.at(i);
  }

  inline const Cell&
  Mesh::cells(const unsigned int local_id) const
  {
    return cells_.at(local_id);
  }
}

#pragma once
#include "framework/mesh/mesh_vector.h"
#include <vector>

#include "cell.h"

namespace pdes
{
  class Mesh;

  class Face final
  {
  public:
    Face() = default;
    Face(const Face&) = default;
    Face(Face&&) noexcept = default;
    virtual ~Face() = default;

    Face& operator=(const Face&) = delete;
    Face& operator=(Face&&) noexcept = delete;

    bool has_neighbor() const { return has_neighbor_; }
    bool is_boundary() const { return not has_neighbor_; }

    void set_neighbor_id(unsigned int neighbor_id);
    unsigned int neighbor_id() const { return neighbor_id_; }

    void set_boundary_id(unsigned int boundary_id);
    unsigned int boundary_id() const { return boundary_id_; }

    const MeshVector<>& centroid() const { return centroid_; }
    const MeshVector<>& normal() const { return normal_; }
    double area() const { return area_; }

    void set_vertex_ids(std::initializer_list<unsigned int> vertex_ids);

    unsigned int vertex_ids(unsigned int vid) const;
    const std::vector<unsigned int>& vertex_ids() const { return vertex_ids_; }

    void compute_geometric_data(const Mesh& mesh) {}

  private:
    bool has_neighbor_ = false;
    unsigned int neighbor_id_ = 0;
    unsigned int boundary_id_ = 0;

    MeshVector<> centroid_;
    MeshVector<> normal_;
    double area_ = 0.0;

    std::vector<unsigned int> vertex_ids_;
  };

  inline void
  Face::set_neighbor_id(const unsigned int neighbor_id)
  {
    has_neighbor_ = true;
    neighbor_id_ = neighbor_id;
  }

  inline void
  Face::set_boundary_id(const unsigned int boundary_id)
  {
    has_neighbor_ = false;
    boundary_id_ = boundary_id;
  }

  inline void
  Face::set_vertex_ids(const std::initializer_list<unsigned int> vertex_ids)
  {
    vertex_ids_ = vertex_ids;
  }

  inline unsigned int
  Face::vertex_ids(const unsigned int vid) const
  {
    return vertex_ids_.at(vid);
  }
}

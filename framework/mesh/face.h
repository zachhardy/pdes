#pragma once
#include "framework/types.h"
#include "framework/mesh/mesh_vector.h"
#include <vector>


namespace pdes
{
  class Mesh;

  class Face final
  {
  public:
    Face() = default;
    Face(const Face&) = default;
    Face(Face&&) noexcept = default;
    ~Face() = default;

    Face& operator=(const Face&) = delete;
    Face& operator=(Face&&) noexcept = delete;

    bool has_neighbor() const { return has_neighbor_; }
    bool is_boundary() const { return not has_neighbor_; }

    void set_neighbor_id(types::global_index neighbor_id);
    types::global_index neighbor_id() const { return neighbor_id_; }

    void set_boundary_id(unsigned int boundary_id);
    unsigned int boundary_id() const { return boundary_id_; }

    const MeshVector<>& centroid() const { return centroid_; }
    const MeshVector<>& normal() const { return normal_; }
    types::real area() const { return area_; }

    void set_vertex_ids(std::vector<types::global_index> vertex_ids);
    types::global_index vertex_ids(unsigned int vid) const;
    const std::vector<types::global_index>& vertex_ids() const;

    void compute_geometric_data(const Mesh& mesh) {}

  private:
    bool has_neighbor_ = false;
    unsigned int neighbor_id_ = 0;
    unsigned int boundary_id_ = 0;

    MeshVector<> centroid_;
    MeshVector<> normal_;
    double area_ = 0.0;

    std::vector<types::global_index> vertex_ids_;
  };
}

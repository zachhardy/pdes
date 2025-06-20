#pragma once
#include "framework/types.h"
#include "framework/mesh/face.h"
#include "framework/mesh/mesh_vector.h"
#include <vector>

namespace pdes
{
  class Mesh;

  class Cell final
  {
  public:
    enum class Type
    {
      SLAB = 1,
      POLYGON = 4,
      TRIANGLE = 5,
      QUADRILATERAL = 6,
    };

    Cell(Type type, Type sub_type);

    Cell(const Cell&) = default;
    Cell(Cell&&) noexcept = default;
    ~Cell() = default;

    Cell& operator=(const Cell&) = delete;
    Cell& operator=(Cell&&) noexcept = delete;

    Type type() const { return type_; }
    Type sub_type() const { return sub_type_; }

    void set_local_id(const unsigned int local_id) { local_id_ = local_id; }
    unsigned int local_id() const { return local_id_; }

    void set_global_id(types::global_index global_id);
    types::global_index global_id() const { return global_id_; }

    void set_partition_id(unsigned int partition_id);
    unsigned int partition_id() const { return partition_id_; }

    void set_block_id(const unsigned int block_id) { block_id_ = block_id; }
    unsigned int block_id() const { return block_id_; }

    const MeshVector<>& centroid() const { return centroid_; }
    types::real volume() const { return volume_; }

    void set_vertex_ids(std::vector<types::global_index>&& vertex_ids);
    types::global_index vertex_ids(unsigned int vid) const;
    const std::vector<types::global_index>& vertex_ids() const;

    void add_face(Face&& face) { faces_.push_back(std::move(face)); }
    void add_faces(std::vector<Face>&& faces);

    Face& faces(const unsigned int f) { return faces_.at(f); }
    const Face& faces(const unsigned int f) const { return faces_.at(f); }
    std::vector<Face>& faces() { return faces_; }
    const std::vector<Face>& faces() const { return faces_; }

    void compute_geometric_info(const Mesh& mesh) {}

  private:
    const Type type_;
    const Type sub_type_;

    unsigned int local_id_ = 0;
    unsigned int global_id_ = 0;
    unsigned int partition_id_ = 0;
    unsigned int block_id_ = 0;

    MeshVector<> centroid_;
    double volume_ = 0.0;

    std::vector<unsigned int> vertex_ids_;
    std::vector<Face> faces_;
  };
}

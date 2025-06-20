#include "framework/mesh/orthomesh_generator.h"

#include "framework/mesh/mesh.h"
#include "framework/mesh/cell.h"
#include "framework/mesh/face.h"
#include "framework/mesh/mesh_vector.h"

namespace pdes
{
  Mesh
  create_1d_orthomesh(const std::vector<types::real>& x_coords,
                      const Mesh::CoordinateSystem coord_sys)
  {
    // Initialize the mesh
    Mesh mesh(1, coord_sys);
    mesh.set_ortho_attributes(x_coords.size() - 1);

    // Create vertices
    for (size_t i = 0; i < x_coords.size(); ++i)
    {
      if (i > 0 and x_coords[i] <= x_coords[i - 1])
        throw std::logic_error("The x-coordinates must be in ascending order.");
      mesh.add_vertex(static_cast<types::global_index>(i),
                      MeshVector(x_coords.at(i)));
    }

    // Define mesh objects
    const auto n_cells = x_coords.size() - 1;
    for (types::global_index c = 0; c < n_cells; ++c)
    {
      Cell cell(Cell::Type::SLAB, Cell::Type::SLAB);
      cell.set_local_id(c);
      cell.set_global_id(c);
      cell.set_vertex_ids({c, c + 1});

      Face left, rite;
      left.set_vertex_ids({c});
      rite.set_vertex_ids({c + 1});

      if (c == 0)
      {
        left.set_boundary_id(0);
        rite.set_neighbor_id(c + 1);
      }
      else if (c == n_cells - 1)
      {
        left.set_neighbor_id(c - 1);
        rite.set_boundary_id(1);
      }
      else
      {
        left.set_neighbor_id(c - 1);
        rite.set_neighbor_id(c + 1);
      }

      cell.add_faces({left, rite});
      cell.compute_geometric_info(mesh);
      mesh.add_cell(std::move(cell));
    }
    return mesh;
  }

  Mesh
  create_1d_orthomesh(const std::vector<types::real>& region_edges,
                      const std::vector<unsigned int>& cells_per_regions,
                      const std::vector<unsigned int>& block_ids,
                      const Mesh::CoordinateSystem coord_sys)
  {
    const auto nr = cells_per_regions.size();

    if (region_edges.size() - 1 != nr)
      throw std::logic_error("Incompatible region edges and cell counts.");

    // Define the vertices on the mesh
    std::vector<types::real> verts;
    for (unsigned int r = 0; r < nr; ++r)
    {
      const auto width = region_edges[r + 1] - region_edges[r];
      const auto dx = width / static_cast<types::real>(cells_per_regions[r]);
      for (unsigned int c = 0; c < cells_per_regions[r]; ++c)
        verts.emplace_back(region_edges[r] + c * dx);
    }
    verts.emplace_back(region_edges.back());

    // Create a mesh from the vertices
    auto mesh = create_1d_orthomesh(verts, coord_sys);

    // Assign block ids to each region
    unsigned int idx = 0;
    for (unsigned int r = 0; r < nr; ++r)
      for (unsigned int c = 0; c < cells_per_regions[r]; ++c)
        mesh.cells(idx++).set_block_id(block_ids[r]);

    return mesh;
  }

  Mesh
  create_2d_orthomesh(const std::vector<types::real>& x_coords,
                      const std::vector<types::real>& y_coords,
                      const Mesh::CoordinateSystem coord_sys)
  {
    Mesh mesh(2, coord_sys);
    mesh.set_ortho_attributes(x_coords.size() - 1, y_coords.size() - 1);

    // Create the vertex/cell mappings (i,j) -> k
    // The ordering is per y-level then per x-level
    types::global_index k = 0;
    const auto nxv = x_coords.size();
    const auto nyv = y_coords.size();
    std::vector vertex_ij_map(nyv, std::vector<unsigned int>(nxv));
    for (unsigned int j = 0; j < nyv; ++j)
      for (unsigned int i = 0; i < nxv; ++i)
      {
        vertex_ij_map[j][i] = k++;
        MeshVector vertex(x_coords.at(i), y_coords.at(j));
        mesh.add_vertex(vertex_ij_map[j][i], std::move(vertex));
      }

    k = 0;
    const auto nxc = nxv - 1;
    const auto nyc = nyv - 1;
    std::vector cell_ij_map(nyc, std::vector<types::global_index>(nxc));
    for (unsigned int j = 0; j < nyc; ++j)
      for (unsigned int i = 0; i < nxc; ++i)
        cell_ij_map[j][i] = k++;

    // Create cells ordered in the same manner. The cell vertices are
    // ordered counter-clockwise from the bottom left vertex.  The faces
    // are ordered similarly from the bottom face.
    for (unsigned int j = 0; j < nyc; ++j)
      for (unsigned int i = 0; i < nxc; ++i)
      {
        // Create cell
        Cell cell(Cell::Type::QUADRILATERAL, Cell::Type::POLYGON);
        cell.set_local_id(cell_ij_map[j][i]);
        cell.set_global_id(cell_ij_map[j][i]);
        cell.set_vertex_ids({
          vertex_ij_map[j][i],
          vertex_ij_map[j][i + 1],
          vertex_ij_map[j + 1][i + 1],
          vertex_ij_map[j + 1][i]
        });

        // Create four faces
        for (int f = 0; f < 4; ++f)
        {
          Face face;

          // The face vertex ids follow the cell vertex ids except
          // for the last face which connects back to the first vertex.
          face.set_vertex_ids({
            cell.vertex_ids(f),
            cell.vertex_ids(f < 3 ? f + 1 : 0)
          });

          // Set default neighbors
          if (f == 0 and j != 0)
            face.set_neighbor_id(cell_ij_map[j - 1][i]);
          else if (f == 1 and i != nxc - 1)
            face.set_neighbor_id(cell_ij_map[j][i + 1]);
          else if (f == 2 and j != nyc - 1)
            face.set_neighbor_id(cell_ij_map[j + 1][i]);
          else if (f == 3 and i != 0)
            face.set_neighbor_id(cell_ij_map[j][i - 1]);

          // Assign boundary flags
          if (f == 0 and j == 0)
            face.set_boundary_id(2);
          else if (f == 1 and i == nxc - 1)
            face.set_neighbor_id(1);
          else if (f == 2 and j == nyc - 1)
            face.set_boundary_id(3);
          else if (f == 3 and j == 0)
            face.set_boundary_id(0);

          cell.add_face(std::move(face));
        } // for face

        cell.compute_geometric_info(mesh);
        mesh.add_cell(std::move(cell));
      } // for i,j cell

    return mesh;
  }

  Mesh
  create_2d_orthomesh(const std::vector<types::real>& x_region_edges,
                      const std::vector<types::real>& y_region_edges,
                      const std::vector<unsigned int>& cells_per_x_region,
                      const std::vector<unsigned int>& cells_per_y_region,
                      const std::vector<unsigned int>& block_ids,
                      const Mesh::CoordinateSystem coord_sys)
  {
    const auto nxr = cells_per_x_region.size();
    const auto nyr = cells_per_y_region.size();

    if (x_region_edges.size() - 1 != nxr)
      throw std::logic_error(
        "Incompatible region edges and cell counts in x-dimension.");
    if (y_region_edges.size() - 1 != nyr)
      throw std::logic_error(
        "Incompatible region edges and cell counts in y-dimension.");
    if (block_ids.size() != nxr * nyr)
      throw std::logic_error("Incompatible number of regions and block IDs.");

    std::vector<types::real> x_verts;
    for (unsigned int r = 0; r < nxr; ++r)
    {
      const auto width = x_region_edges[r + 1] - x_region_edges[r];
      const auto dx = width / static_cast<types::real>(cells_per_x_region[r]);
      for (unsigned int c = 0; c < cells_per_x_region[r]; ++c)
        x_verts.emplace_back(x_region_edges[r] + c * dx);
    }
    x_verts.emplace_back(x_region_edges.back());

    std::vector<types::real> y_verts;
    for (unsigned int r = 0; r < nyr; ++r)
    {
      const auto width = y_region_edges[r + 1] - y_region_edges[r];
      const auto dy = width / static_cast<types::real>(cells_per_y_region[r]);
      for (unsigned int c = 0; c < cells_per_y_region[r]; ++c)
        y_verts.emplace_back(y_region_edges[r] + c * dy);
    }
    y_verts.emplace_back(y_region_edges.back());

    // Create a mesh from the verticles
    auto mesh = create_2d_orthomesh(x_verts, y_verts, coord_sys);

    // Assign block ids to each region
    unsigned int idx = 0;
    for (unsigned int ry = 0; ry < nyr; ++ry)
      for (unsigned int cy = 0; cy < cells_per_y_region[ry]; ++cy)
        for (unsigned int rx = 0; rx < nxr; ++rx)
          for (unsigned int cx = 0; cx < cells_per_x_region[rx]; ++cx)
            mesh.cells(idx++).set_block_id(block_ids[ry * nxr + rx]);

    return mesh;
  }

  Mesh
  create_3d_orthomesh(const std::vector<types::real>& x_coords,
                      const std::vector<types::real>& y_coords,
                      const std::vector<types::real>& z_coords,
                      const Mesh::CoordinateSystem coord_sys)
  {
    Mesh mesh(3, coord_sys);
    mesh.set_ortho_attributes(x_coords.size() - 1,
                              y_coords.size() - 1,
                              z_coords.size() - 1);

    // Create the vertex and cell mappings (i,j, k) -> l
    // The ordering is per level z, per level y, per level x
    types::global_index l = 0;
    const auto nxv = x_coords.size();
    const auto nyv = y_coords.size();
    const auto nzv = z_coords.size();

    std::vector vertex_ijk_map(
      nzv, std::vector(
        nyv, std::vector<types::global_index>(nxv, 0)));

    for (unsigned int k = 0; k < nzv; ++k)
      for (unsigned int j = 0; j < nyv; ++j)
        for (unsigned int i = 0; i < nxv; ++i)
        {
          vertex_ijk_map[k][j][i] = l++;
          MeshVector vertex(x_coords.at(i), y_coords.at(j), z_coords.at(k));
          mesh.add_vertex(vertex_ijk_map[k][j][i], std::move(vertex));
        }

    // Create a mapping for cells from (i,j) -> k
    l = 0;
    const auto nxc = nxv - 1;
    const auto nyc = nyv - 1;
    const auto nzc = nzv - 1;

    std::vector cell_ijk_map(
      nyc, std::vector(
        nxc, std::vector<types::global_index>(nzc, 0)));

    for (unsigned int k = 0; k < nzc; ++k)
      for (unsigned int j = 0; j < nyc; ++j)
        for (unsigned int i = 0; i < nxc; ++i)
          cell_ijk_map[k][j][i] = l++;

    // Create cells ordered in the same manner. The cell vertices are
    // ordered counter-clockwise from the bottom left vertex.  The faces
    // are ordered similarly from the bottom face.
    for (unsigned int j = 0; j < nyc; ++j)
      for (unsigned int i = 0; i < nxc; ++i)
        for (unsigned int k = 0; k < nzc; ++k)
        {
          // Create cell
          Cell cell(Cell::Type::QUADRILATERAL, Cell::Type::POLYGON);
          cell.set_local_id(cell_ijk_map[j][i][k]);
          cell.set_global_id(cell_ijk_map[j][i][k]);
          cell.set_vertex_ids({
            vertex_ijk_map[k][j][i],
            vertex_ijk_map[k][j][i + 1],
            vertex_ijk_map[k][j + 1][i + 1],
            vertex_ijk_map[k][j + 1][i],
            vertex_ijk_map[k + 1][j][i],
            vertex_ijk_map[k + 1][j][i + 1],
            vertex_ijk_map[k + 1][j + 1][i + 1],
            vertex_ijk_map[k + 1][j + 1][i],
          });

          // Left face
          {
            Face face;

            face.set_vertex_ids({
              vertex_ijk_map[k][j][i],
              vertex_ijk_map[k + 1][j][i],
              vertex_ijk_map[k + 1][j + 1][i],
              vertex_ijk_map[k][j + 1][i]
            });

            if (i > 0)
              face.set_neighbor_id(cell_ijk_map[k][j][i - 1]);
            else
              face.set_boundary_id(0);

            cell.add_face(std::move(face));
          }

          // Right face
          {
            Face face;

            face.set_vertex_ids({
              vertex_ijk_map[k][j][i + 1],
              vertex_ijk_map[k][j + 1][i + 1],
              vertex_ijk_map[k + 1][j + 1][i + 1],
              vertex_ijk_map[k + 1][j][i + 1],
            });

            if (i < nxc - 1)
              face.set_neighbor_id(cell_ijk_map[k][j][i + 1]);
            else
              face.set_boundary_id(1);

            cell.add_face(std::move(face));
          }

          // Bottom face
          {
            Face face;

            face.set_vertex_ids({
              vertex_ijk_map[k][j][i],
              vertex_ijk_map[k + 1][j][i],
              vertex_ijk_map[k + 1][j][i + 1],
              vertex_ijk_map[k][j][i + 1]
            });

            if (j > 0)
              face.set_neighbor_id(cell_ijk_map[k][j - 1][i]);
            else
              face.set_boundary_id(2);

            cell.add_face(std::move(face));
          }

          // Top face
          {
            Face face;

            face.set_vertex_ids({
              vertex_ijk_map[k][j + 1][i],
              vertex_ijk_map[k + 1][j + 1][i],
              vertex_ijk_map[k + 1][j + 1][i + 1],
              vertex_ijk_map[k][j + 1][i + 1]
            });

            if (j < nyc - 1)
              face.set_neighbor_id(cell_ijk_map[k][j + 1][i]);
            else
              face.set_boundary_id(3);

            cell.add_face(std::move(face));
          }

          // Back face
          {
            Face face;

            face.set_vertex_ids({
              vertex_ijk_map[k][j][i],
              vertex_ijk_map[k][j + 1][i],
              vertex_ijk_map[k][j + 1][i + 1],
              vertex_ijk_map[k][j][i + 1],
            });

            if (k > 0)
              face.set_neighbor_id(cell_ijk_map[k - 1][j][i]);
            else
              face.set_boundary_id(4);

            cell.add_face(std::move(face));
          }

          // Front face
          {
            Face face;

            face.set_vertex_ids({
              vertex_ijk_map[k + 1][j][i],
              vertex_ijk_map[k + 1][j][i + 1],
              vertex_ijk_map[k + 1][j + 1][i + 1],
              vertex_ijk_map[k + 1][j + 1][i]
            });

            if (k < nxc - 1)
              face.set_neighbor_id(cell_ijk_map[k + 1][i][j]);
            else
              face.set_boundary_id(5);

            cell.add_face(std::move(face));
          }

          cell.compute_geometric_info(mesh);
          mesh.add_cell(std::move(cell));
        } // for i,j,k cell

    return mesh;
  }

  Mesh
  create_3d_orthomesh(const std::vector<types::real>& x_region_edges,
                      const std::vector<types::real>& y_region_edges,
                      const std::vector<types::real>& z_region_edges,
                      const std::vector<unsigned int>& cells_per_x_region,
                      const std::vector<unsigned int>& cells_per_y_region,
                      const std::vector<unsigned int>& cells_per_z_region,
                      const std::vector<unsigned int>& block_ids,
                      const Mesh::CoordinateSystem coord_sys)
  {
    const auto nxr = cells_per_x_region.size();
    const auto nyr = cells_per_y_region.size();
    const auto nzr = cells_per_z_region.size();

    if (x_region_edges.size() - 1 != nxr)
      throw std::logic_error(
        "Incompatible region edges and cell counts in x-dimension.");
    if (y_region_edges.size() - 1 != nyr)
      throw std::logic_error(
        "Incompatible region edges and cell counts in y-dimension.");
    if (z_region_edges.size() - 1 != nzr)
      throw std::logic_error(
        "Incompatible region edges and cell counts in z-dimension.");
    if (block_ids.size() != nxr * nyr * nzr)
      throw std::logic_error("Incompatible number of regions and block IDs.");

    std::vector<types::real> x_verts;
    for (unsigned int r = 0; r < nxr; ++r)
    {
      const auto width = x_region_edges[r + 1] - x_region_edges[r];
      const auto dx = width / static_cast<types::real>(cells_per_x_region[r]);
      for (unsigned int c = 0; c < cells_per_x_region[r]; ++c)
        x_verts.emplace_back(x_region_edges[r] + c * dx);
    }
    x_verts.emplace_back(x_region_edges.back());

    std::vector<types::real> y_verts;
    for (unsigned int r = 0; r < nyr; ++r)
    {
      const auto width = y_region_edges[r + 1] - y_region_edges[r];
      const auto dy = width / static_cast<types::real>(cells_per_y_region[r]);
      for (unsigned int c = 0; c < cells_per_y_region[r]; ++c)
        y_verts.emplace_back(y_region_edges[r] + c * dy);
    }
    y_verts.emplace_back(y_region_edges.back());

    std::vector<types::real> z_verts;
    for (unsigned int r = 0; r < nzr; ++r)
    {
      const auto width = z_region_edges[r + 1] - z_region_edges[r];
      const auto dz = width / static_cast<types::real>(cells_per_z_region[r]);
      for (unsigned int c = 0; c < cells_per_y_region[r]; ++c)
        y_verts.emplace_back(z_region_edges[r] + c * dz);
    }
    z_verts.emplace_back(z_region_edges.back());

    // Create a mesh from the verticles
    auto mesh = create_3d_orthomesh(x_verts, y_verts, z_verts, coord_sys);

    // Assign block ids to each region
    unsigned int idx = 0;
    for (unsigned int rz = 0; rz < nzr; ++rz)
      for (unsigned int cz = 0; cz < cells_per_z_region[rz]; ++cz)
        for (unsigned int ry = 0; ry < nyr; ++ry)
          for (unsigned int cy = 0; cy < cells_per_y_region[ry]; ++cy)
            for (unsigned int rx = 0; rx < nxr; ++rx)
              for (unsigned int cx = 0; cx < cells_per_x_region[rx]; ++cx)
              {
                const auto bid = rz * nxr * nyr + ry * nyr + rx;
                mesh.cells(idx++).set_block_id(block_ids[bid]);
              }

    return mesh;
  }
}

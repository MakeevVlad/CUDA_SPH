
#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "../info.cuh"
#include <vector>
#include <iostream>
#include <string>
#include <utility>
#include "../particle.cuh"

using index_t = std::size_t;


// Material class, thhat store all information of materials in initial
// distribution. Will be used to form particles from mesh, so need to
// handle all your additional data related to particles
class Material {
 private:
  real_t density;

  // tag and name -- values gotten from mesh
  std::string name;
  int tag;
 public:
  Material(std::string name, int tag) : density(0), name(name), tag(tag) {};
  // void set_parametres(float dens) { density = dens; };
  int get_tag() { return tag; };

  // We almost always need density
  real_t get_density() { return density; };

  // handler of line in .mat file (physical tag will be parsed in
  // VolumeMesh::construct_materials() is not included)
  void set_parametres(std::istream& input) {
    input >> density;
  };
};


class Point {
 private:
  std::vector<real_t> x;
 public:
  Point();
  Point(real_t, real_t, real_t);
  real_t& operator[](int i);
  std::istream& operator>>(std::istream& in);
};

class Triangle {
 private:
  std::vector<index_t> points;
 public:
  Triangle();
  Triangle(index_t p1, index_t p2, index_t p3);

  std::istream& operator>>(std::istream& in);
  index_t& operator[](int i);
};

class Tetrahedra {
 private:
  std::vector<index_t> points;
  index_t material;
 public:
  Tetrahedra();
  Tetrahedra(index_t p1, index_t p2, index_t p3, index_t p4, index_t material);
  index_t& operator[](int i);
  index_t get_material_id();
};

class SurfaceMesh {
 private:
  std::vector<Point> pts;
  std::vector<Triangle> trs;
 public:
  SurfaceMesh();
  void construct_from_file(std::string fname);


  // An algorithm of work - get triangles count N, 
  // prepare triangles_array of size N with initialized
  // memory and give this array to get_alll_triangles function with N
  index_t get_triangles_count();
  void get_triangle_as_array(real_t** triangle, index_t id);
  void get_all_triangles(real_t*** triangles_array, index_t count);
};

class VolumeMesh {
 private:
  std::vector<Point> pts;
  std::vector<Tetrahedra> tetrs;
  std::vector<Material> mats;
 public:
  VolumeMesh();
  void construct_from_file(std::string fmesh, std::string fmat);
  void construct_materials(std::string fmat);
  // TODO : generate_particles

  std::vector<real_t> get_mass_center(index_t tetr);
  real_t get_representative_sphere_radius(index_t tetr);
  real_t get_volume(index_t tetr);
  real_t get_mass(index_t tetr);

  std::size_t get_tetrahedra_number();

  void initialize_particle(Particle* part, index_t tetr);
};

#endif
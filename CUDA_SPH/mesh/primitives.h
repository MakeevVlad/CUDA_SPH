
#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "../info.cuh"
#include <vector>
#include <iostream>
#include <string>

using index_t = std::size_t;


class Material {
 private:
  float density;
  std::string name;
  int tag;
 public:
  Material(std::string name, int tag) : density(0), name(name), tag(tag) {};
  void set_parametres(float dens) { density = dens; };
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
  // TODO : generate_particles
};

#endif
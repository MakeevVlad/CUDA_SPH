
#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "../info.cuh"
#include <vector>
#include <iostream>
#include <string>

using index_t = std::size_t;

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

#endif
#include "redner.h"
#include "py_utils.h"
#include "pathtracer.h"
#include "camera.h"
#include "scene.h"
#include "shape.h"
#include "material.h"
#include "area_light.h"
#include "envmap.h"
#include "active_pixels.h"
#include "ptr.h"
#include "load_serialized.h"

#include <pybind11/stl.h>

PYBIND11_MODULE(redner, m) {
    m.doc() = "Redner"; // optional module docstring

    py::class_<ptr<float>>(m, "float_ptr")
        .def(py::init<std::size_t>());
    py::class_<ptr<int>>(m, "int_ptr")
        .def(py::init<std::size_t>());

    py::class_<Camera>(m, "Camera")
        .def(py::init<int,
                      int,
                      ptr<float>,
                      ptr<float>,
                      ptr<float>,
                      ptr<float>,
                      float,
                      bool>());

    py::class_<DCamera>(m, "DCamera")
        .def(py::init<ptr<float>,
                      ptr<float>,
                      ptr<float>,
                      ptr<float>>());

    py::class_<Scene>(m, "Scene")
        .def(py::init<const Camera &,
                      const std::vector<const Shape*> &,
                      const std::vector<const Material*> &,
                      const std::vector<const AreaLight*> &,
                      const std::shared_ptr<const EnvironmentMap> &,
                      bool,
                      int>());

    py::class_<DScene, std::shared_ptr<DScene>>(m, "DScene")
        .def(py::init<const DCamera &,
                      const std::vector<DShape*> &,
                      const std::vector<DMaterial*> &,
                      const std::vector<DAreaLight*> &,
                      const std::shared_ptr<DEnvironmentMap> &,
                      bool,
                      int>());

    py::class_<Shape>(m, "Shape")
        .def(py::init<ptr<float>,
                      ptr<int>,
                      ptr<float>,
                      ptr<float>,
                      int,
                      int,
                      int,
                      int>())
        .def_readonly("num_vertices", &Shape::num_vertices)
        .def("has_uvs", &Shape::has_uvs)
        .def("has_normals", &Shape::has_normals);

    py::class_<DShape>(m, "DShape")
        .def(py::init<ptr<float>,
                      ptr<float>,
                      ptr<float>>());

    py::class_<Texture1>(m, "Texture1")
        .def(py::init<ptr<float>,
                      int,
                      int,
                      int,
                      ptr<float>>());

    py::class_<Texture3>(m, "Texture3")
        .def(py::init<ptr<float>,
                      int,
                      int,
                      int,
                      ptr<float>>());

    py::class_<Material>(m, "Material")
        .def(py::init<Texture3,
                      Texture3,
                      Texture1,
                      bool>())
        .def("get_diffuse_size", &Material::get_diffuse_size)
        .def("get_specular_size", &Material::get_specular_size)
        .def("get_roughness_size", &Material::get_roughness_size);

    py::class_<DMaterial>(m, "DMaterial")
        .def(py::init<Texture3,
                      Texture3,
                      Texture1>());

    py::class_<AreaLight>(m, "AreaLight")
        .def(py::init<int,
                      ptr<float>,
                      bool>());

    py::class_<DAreaLight>(m, "DAreaLight")
        .def(py::init<ptr<float>>());

    py::class_<EnvironmentMap, std::shared_ptr<EnvironmentMap>>(m, "EnvironmentMap")
        .def(py::init<Texture3,   // values
                      ptr<float>, // env_to_world
                      ptr<float>, // world_to_env
                      ptr<float>, // sample_cdf_ys
                      ptr<float>, // sample_cdf_xs
                      Real>())
        .def("get_size", &EnvironmentMap::get_size);
    py::class_<DEnvironmentMap, std::shared_ptr<DEnvironmentMap>>(m, "DEnvironmentMap")
        .def(py::init<Texture3,       // values
                      ptr<float>>()); // world_to_env

    py::enum_<Channels>(m, "channels")
        .value("radiance", Channels::radiance)
        .value("alpha", Channels::alpha)
        .value("depth", Channels::depth)
        .value("position", Channels::position)
        .value("geometry_normal", Channels::geometry_normal)
        .value("shading_normal", Channels::shading_normal)
        .value("uv", Channels::uv)
        .value("diffuse_reflectance", Channels::diffuse_reflectance)
        .value("specular_reflectance", Channels::specular_reflectance)
        .value("roughness", Channels::roughness)
        .value("shape_id", Channels::shape_id)
        .value("material_id", Channels::material_id);

    m.def("compute_num_channels", compute_num_channels, "");

    py::class_<RenderOptions>(m, "RenderOptions")
        .def(py::init<uint64_t,
                      int,
                      int,
                      std::vector<Channels>>())
        .def_readwrite("seed", &RenderOptions::seed)
        .def_readwrite("num_samples", &RenderOptions::num_samples);

    py::class_<Vector2f>(m, "Vector2f")
        .def_readwrite("x", &Vector2f::x)
        .def_readwrite("y", &Vector2f::y);

    py::class_<Vector3f>(m, "Vector3f")
        .def_readwrite("x", &Vector3f::x)
        .def_readwrite("y", &Vector3f::y)
        .def_readwrite("z", &Vector3f::z);

    py::class_<MitsubaTriMesh>(m, "MitsubaTriMesh")
        .def_readwrite("vertices", &MitsubaTriMesh::vertices)
        .def_readwrite("indices", &MitsubaTriMesh::indices)
        .def_readwrite("uvs", &MitsubaTriMesh::uvs)
        .def_readwrite("normals", &MitsubaTriMesh::normals);

    m.def("load_serialized", &load_serialized, "");
    m.def("render", &render, "");

    /// Tests
    m.def("test_sample_primary_rays", &test_sample_primary_rays, "");
    m.def("test_scene_intersect", &test_scene_intersect, "");
    m.def("test_sample_point_on_light", &test_sample_point_on_light, "");
    m.def("test_active_pixels", &test_active_pixels, "");
    m.def("test_camera_derivatives", &test_camera_derivatives, "");
    m.def("test_d_bsdf", &test_d_bsdf, "");
    m.def("test_d_bsdf_sample", &test_d_bsdf_sample, "");
    m.def("test_d_bsdf_pdf", &test_d_bsdf_pdf, "");
    m.def("test_d_intersect", &test_d_intersect, "");
    m.def("test_d_sample_shape", &test_d_sample_shape, "");
}

#include "redner.h"
#include "active_pixels.h"
#include "area_light.h"
#include "automatic_uv_map.h"
#include "camera.h"
#include "camera_distortion.h"
#include "envmap.h"
#include "load_serialized.h"
#include "material.h"
#include "pathtracer.h"
#include "ptr.h"
#include "scene.h"
#include "shape.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(redner, m) {
    m.doc() = "Redner"; // optional module docstring

    py::class_<ptr<float>>(m, "float_ptr")
        .def(py::init<std::size_t>());
    py::class_<ptr<int>>(m, "int_ptr")
        .def(py::init<std::size_t>());

    py::enum_<CameraType>(m, "CameraType")
        .value("perspective", CameraType::Perspective)
        .value("orthographic", CameraType::Orthographic)
        .value("fisheye", CameraType::Fisheye)
        .value("panorama", CameraType::Panorama);

    py::class_<Camera>(m, "Camera")
        .def(py::init<int,
                      int,
                      ptr<float>, // position
                      ptr<float>, // look
                      ptr<float>, // up
                      ptr<float>, // cam_to_world
                      ptr<float>, // world_to_cam
                      ptr<float>, // ndc_to_cam
                      ptr<float>, // cam_to_ndc
                      ptr<float>, // distortion_params
                      float, // clip_near
                      CameraType,
                      Vector2i, // viewport_beg
                      Vector2i>()) // viewport_end
        .def_readonly("use_look_at", &Camera::use_look_at)
        .def("has_distortion_params", &Camera::has_distortion_params);

    py::class_<DCamera>(m, "DCamera")
        .def(py::init<ptr<float>, // position
                      ptr<float>, // look
                      ptr<float>, // up
                      ptr<float>, // cam_to_world
                      ptr<float>, // world_to_cam
                      ptr<float>, // ndc_to_cam
                      ptr<float>, // cam_to_ndc
                      ptr<float>>()); // distortion_params

    py::class_<Scene>(m, "Scene")
        .def(py::init<const Camera &,
                      const std::vector<const Shape*> &,
                      const std::vector<const Material*> &,
                      const std::vector<const AreaLight*> &,
                      const std::shared_ptr<const EnvironmentMap> &,
                      bool,
                      int,
                      bool,
                      bool>())
        .def_readonly("max_generic_texture_dimension",
            &Scene::max_generic_texture_dimension);

    py::class_<DScene, std::shared_ptr<DScene>>(m, "DScene")
        .def(py::init<const DCamera &,
                      const std::vector<DShape*> &,
                      const std::vector<DMaterial*> &,
                      const std::vector<DAreaLight*> &,
                      const std::shared_ptr<DEnvironmentMap> &,
                      bool,
                      int>());

    py::class_<Shape>(m, "Shape")
        .def(py::init<ptr<float>, // vertices
                      ptr<int>, // indices
                      ptr<float>, // uvs
                      ptr<float>, // normals
                      ptr<int>, // uv_indices
                      ptr<int>, // normal_indices
                      ptr<float>, // colors
                      int, // num_vertices
                      int, // num_uv_vertices
                      int, // num_normal_vertices
                      int, // num_triangles
                      int, // material_id
                      int  // light_id
                      >())
        .def_readonly("num_vertices", &Shape::num_vertices)
        .def_readonly("num_uv_vertices", &Shape::num_uv_vertices)
        .def_readonly("num_normal_vertices", &Shape::num_normal_vertices)
        .def("has_uvs", &Shape::has_uvs)
        .def("has_normals", &Shape::has_normals)
        .def("has_colors", &Shape::has_colors);

    py::class_<DShape>(m, "DShape")
        .def(py::init<ptr<float>,
                      ptr<float>,
                      ptr<float>,
                      ptr<float>>());

    py::class_<Texture1>(m, "Texture1")
        .def(py::init<const std::vector<ptr<float>> &,
                      const std::vector<int> &, // width
                      const std::vector<int> &, // height
                      int, // channels
                      ptr<float>>());

    py::class_<Texture3>(m, "Texture3")
        .def(py::init<const std::vector<ptr<float>> &,
                      const std::vector<int> &, // width
                      const std::vector<int> &, // height
                      int, // channels
                      ptr<float>>());

    py::class_<TextureN>(m, "TextureN")
        .def(py::init<const std::vector<ptr<float>> &,
                      const std::vector<int> &, // width
                      const std::vector<int> &, // height
                      int, // channels
                      ptr<float>>());

    py::class_<Material>(m, "Material")
        .def(py::init<Texture3, // diffuse
                      Texture3, // specular
                      Texture1, // roughness
                      TextureN, // generic_texture
                      Texture3, // normal_map
                      bool, // compute_specular_lighting
                      bool, // two_sided
                      bool>()) // use_vertex_color
        .def("get_diffuse_levels", &Material::get_diffuse_levels)
        .def("get_diffuse_size", &Material::get_diffuse_size)
        .def("get_specular_levels", &Material::get_specular_levels)
        .def("get_specular_size", &Material::get_specular_size)
        .def("get_roughness_levels", &Material::get_roughness_levels)
        .def("get_roughness_size", &Material::get_roughness_size)
        .def("get_generic_levels", &Material::get_generic_levels)
        .def("get_generic_size", &Material::get_generic_size)
        .def("get_normal_map_levels", &Material::get_normal_map_levels)
        .def("get_normal_map_size", &Material::get_normal_map_size);

    py::class_<DMaterial>(m, "DMaterial")
        .def(py::init<Texture3, // diffuse
                      Texture3, // specular
                      Texture1, // roughness
                      TextureN, // generic_texture
                      Texture3>()); // normal_map

    py::class_<AreaLight>(m, "AreaLight")
        .def(py::init<int, // shape_id
                      ptr<float>, // intensity
                      bool, // two_sided
                      bool>()); // directly_visible

    py::class_<DAreaLight>(m, "DAreaLight")
        .def(py::init<ptr<float>>());

    py::class_<EnvironmentMap, std::shared_ptr<EnvironmentMap>>(m, "EnvironmentMap")
        .def(py::init<Texture3,   // values
                      ptr<float>, // env_to_world
                      ptr<float>, // world_to_env
                      ptr<float>, // sample_cdf_ys
                      ptr<float>, // sample_cdf_xs
                      Real, // pdf_norm
                      bool>()) // directly_visible
        .def("get_levels", &EnvironmentMap::get_levels)
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
        .value("barycentric_coordinates", Channels::barycentric_coordinates)
        .value("diffuse_reflectance", Channels::diffuse_reflectance)
        .value("specular_reflectance", Channels::specular_reflectance)
        .value("roughness", Channels::roughness)
        .value("generic_texture", Channels::generic_texture)
        .value("vertex_color", Channels::vertex_color)
        .value("shape_id", Channels::shape_id)
        .value("triangle_id", Channels::triangle_id)
        .value("material_id", Channels::material_id);

    m.def("compute_num_channels", compute_num_channels, "");

    py::enum_<SamplerType>(m, "SamplerType")
        .value("independent", SamplerType::independent)
        .value("sobol", SamplerType::sobol);

    py::class_<RenderOptions>(m, "RenderOptions")
        .def(py::init<uint64_t,
                      int, // num_samples
                      int, // max_bounces
                      std::vector<Channels>,
                      SamplerType,
                      bool // sample_pixel_center
                      >())
        .def_readwrite("seed", &RenderOptions::seed)
        .def_readwrite("num_samples", &RenderOptions::num_samples);

    py::class_<Vector2i>(m, "Vector2i")
        .def(py::init<int, int>())
        .def_readwrite("x", &Vector2i::x)
        .def_readwrite("y", &Vector2i::y);

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

    // For auto uv unwrapping
    py::class_<UVTriMesh>(m, "UVTriMesh")
        .def(py::init<ptr<float>, // vertices
                      ptr<int>, // indices
                      ptr<float>, // uvs
                      ptr<int>, // uv_indices
                      int, // num_vertices
                      int, // num_uv_vertices
                      int>()) // num_triangles
        .def_readwrite("uvs", &UVTriMesh::uvs)
        .def_readwrite("uv_indices", &UVTriMesh::uv_indices)
        .def_readwrite("num_uv_vertices", &UVTriMesh::num_uv_vertices);
    py::class_<TextureAtlas>(m, "TextureAtlas")
        .def(py::init<>());
    m.def("automatic_uv_map", &automatic_uv_map, "");
    m.def("copy_texture_atlas", &copy_texture_atlas, "");

    m.def("render", &render, "");

    /// Tests
    m.def("test_sample_primary_rays", &test_sample_primary_rays, "");
    m.def("test_scene_intersect", &test_scene_intersect, "");
    m.def("test_sample_point_on_light", &test_sample_point_on_light, "");
    m.def("test_active_pixels", &test_active_pixels, "");
    m.def("test_camera_derivatives", &test_camera_derivatives, "");
    m.def("test_camera_distortion", &test_camera_distortion, "");
    m.def("test_d_bsdf", &test_d_bsdf, "");
    m.def("test_d_bsdf_sample", &test_d_bsdf_sample, "");
    m.def("test_d_bsdf_pdf", &test_d_bsdf_pdf, "");
    m.def("test_d_intersect", &test_d_intersect, "");
    m.def("test_d_sample_shape", &test_d_sample_shape, "");
    m.def("test_atomic", &test_atomic, "");
}

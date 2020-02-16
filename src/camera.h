#pragma once

#include "redner.h"
#include "vector.h"
#include "buffer.h"
#include "ray.h"
#include "transform.h"
#include "ptr.h"
#include "atomic.h"

enum class CameraType {
    Perspective,
    Orthographic,
    Fisheye,
    Panorama
};

struct Camera {
    Camera() {}

    Camera(int width,
           int height,
           ptr<float> position_,
           ptr<float> look_,
           ptr<float> up_,
           ptr<float> cam_to_world_,
           ptr<float> world_to_cam_,
           ptr<float> intrinsic_mat_inv,
           ptr<float> intrinsic_mat,
           float clip_near,
           CameraType camera_type)
        : width(width),
          height(height),
          intrinsic_mat_inv(intrinsic_mat_inv.get()),
          intrinsic_mat(intrinsic_mat.get()),
          clip_near(clip_near),
          camera_type(camera_type) {
        if (cam_to_world_.get()) {
            cam_to_world = Matrix4x4(cam_to_world_.get());
            world_to_cam = Matrix4x4(world_to_cam_.get());
            use_look_at = false;
        } else {
            position = Vector3{position_[0], position_[1], position_[2]};
            look = Vector3{look_[0], look_[1], look_[2]};
            up = Vector3{up_[0], up_[1], up_[2]};
            cam_to_world = look_at_matrix(position, look, up);
            world_to_cam = inverse(cam_to_world);
            use_look_at = true;
        }
    }

    int width, height;
    bool use_look_at;
    Vector3 position, look, up;
    Matrix4x4 cam_to_world;
    Matrix4x4 world_to_cam;
    Matrix3x3 intrinsic_mat_inv;
    Matrix3x3 intrinsic_mat;
    float clip_near;
    CameraType camera_type;
};

struct DCamera {
    DCamera() {}
    DCamera(ptr<float> position,
            ptr<float> look,
            ptr<float> up,
            ptr<float> cam_to_world,
            ptr<float> world_to_cam,
            ptr<float> intrinsic_mat_inv,
            ptr<float> intrinsic_mat)
        : position(position.get()),
          look(look.get()),
          up(up.get()),
          cam_to_world(cam_to_world.get()),
          world_to_cam(world_to_cam.get()),
          intrinsic_mat_inv(intrinsic_mat_inv.get()),
          intrinsic_mat(intrinsic_mat.get()) {}

    float *position;
    float *look;
    float *up;
    float *cam_to_world;
    float *world_to_cam;
    float *intrinsic_mat_inv;
    float *intrinsic_mat;
};

template <typename T>
struct TCameraSample {
    TVector2<T> xy;
};

using CameraSample = TCameraSample<Real>;

DEVICE
inline
Ray sample_primary(const Camera &camera,
                   const Vector2 &screen_pos) {
    switch(camera.camera_type) {
        case CameraType::Perspective: {
            // Linear projection
            auto org = xfm_point(camera.cam_to_world, Vector3{0, 0, 0});
            // [0, 1] x [0, 1] -> [-1, 1/aspect_ratio] x [1, -1/aspect_ratio]
            auto aspect_ratio = Real(camera.width) / Real(camera.height);
            auto pt = Vector3{(screen_pos[0] - 0.5f) * 2.f,
                              (screen_pos[1] - 0.5f) * (-2.f) / aspect_ratio,
                              Real(1)};
            auto dir = camera.intrinsic_mat_inv * pt;
            auto n_dir = normalize(dir);
            auto world_dir = xfm_vector(camera.cam_to_world, n_dir);
            auto n_world_dir = normalize(world_dir);
            return Ray{org, n_world_dir};
        }
        case CameraType::Orthographic: {
            // Linear projection
            // [0, 1] x [0, 1] -> [-1, 1/aspect_ratio] x [1, -1/aspect_ratio]
            auto aspect_ratio = Real(camera.width) / Real(camera.height);
            auto pt = Vector3{(screen_pos[0] - 0.5f) * 2.f,
                              (screen_pos[1] - 0.5f) * (-2.f) / aspect_ratio,
                              Real(0)};
            auto org = xfm_point(camera.cam_to_world, camera.intrinsic_mat_inv * pt);
            auto dir = xfm_vector(camera.cam_to_world, Vector3{0, 0, 1});
            auto n_dir = normalize(dir);
            return Ray{org, n_dir};
        }
        case CameraType::Fisheye: {
            // Equi-angular projection
            auto org = xfm_point(camera.cam_to_world, Vector3{0, 0, 0});
            // x, y to polar coordinate
            auto x = 2.f * (screen_pos.x - 0.5f);
            auto y = 2.f * (screen_pos.y - 0.5f);
            if (x * x + y * y > 1.f) {
                return Ray{Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            }
            auto r = sqrt(x*x + y*y);
            auto phi = atan2(y, x);
            // polar coordinate to spherical, map r to angle through polynomial
            auto theta = r * Real(M_PI / 2);
            auto sin_phi = sin(phi);
            auto cos_phi = cos(phi);
            auto sin_theta = sin(theta);
            auto cos_theta = cos(theta);
            auto dir = Vector3{-cos_phi * sin_theta, -sin_phi * sin_theta, cos_theta};
            auto world_dir = xfm_vector(camera.cam_to_world, dir);
            auto n_world_dir = normalize(world_dir);
            return Ray{org, n_world_dir};
        }
        case CameraType::Panorama: {
            auto org = xfm_point(camera.cam_to_world, Vector3{0, 0, 0});
            // x, y to spherical coordinate
            auto theta = Real(M_PI) * screen_pos.y;
            auto phi = Real(2 * M_PI) * screen_pos.x;
            auto sin_phi = sin(phi);
            auto cos_phi = cos(phi);
            auto sin_theta = sin(theta);
            auto cos_theta = cos(theta);
            auto dir = Vector3{cos_phi * sin_theta,
                               cos_theta,
                               sin_phi * sin_theta};
            auto world_dir = xfm_vector(camera.cam_to_world, dir);
            auto n_world_dir = normalize(world_dir);
            return Ray{org, n_world_dir};
        }
        default: {
            assert(false);
            return Ray{};
        }
    }
}

DEVICE
inline void d_sample_primary_ray(const Camera &camera,
                                 const Vector2 &screen_pos,
                                 const DRay &d_ray,
                                 DCamera &d_camera,
                                 Vector2 *d_screen_pos) {
    switch(camera.camera_type) {
        case CameraType::Perspective: {
            // Linear projection
            // auto org = xfm_point(camera.cam_to_world, Vector3{0, 0, 0});
            // [0, 1] x [0, 1] -> [-1, 1/aspect_ratio] x [1, -1/aspect_ratio]
            auto aspect_ratio = Real(camera.width) / Real(camera.height);
            auto pt = Vector3{(screen_pos[0] - 0.5f) * 2.f,
                               (screen_pos[1] - 0.5f) * (-2.f) / aspect_ratio,
                               Real(1)};
            // Assume film at z=1, thus w=tan(fov), h=tan(fov) / aspect_ratio
            auto dir = camera.intrinsic_mat_inv * pt;
            auto n_dir = normalize(dir);
            auto world_dir = xfm_vector(camera.cam_to_world, n_dir);
            // auto n_world_dir = normalize(world_dir);

            // ray = Ray{org, n_world_dir};
            auto d_org = d_ray.org;
            auto d_n_world_dir = d_ray.dir;
            auto d_world_dir = d_normalize(world_dir, d_n_world_dir);
            auto d_n_dir = Vector3{0, 0, 0};
            // world_dir = xfm_vector(camera.cam_to_world, n_dir)
            auto d_cam_to_world = Matrix4x4();
            d_xfm_vector(camera.cam_to_world, n_dir, d_world_dir,
                         d_cam_to_world, d_n_dir);
            // n_dir = normalize(dir)
            auto d_dir = d_normalize(dir, d_n_dir);
            // dir = camera.intrinsic_mat_inv * pt
            auto d_intrinsic_mat_inv = Matrix3x3{};
            d_intrinsic_mat_inv(0, 0) += d_dir[0] * pt[0];
            d_intrinsic_mat_inv(0, 1) += d_dir[0] * pt[1];
            d_intrinsic_mat_inv(0, 2) += d_dir[0] * pt[2];
            d_intrinsic_mat_inv(1, 0) += d_dir[1] * pt[0];
            d_intrinsic_mat_inv(1, 1) += d_dir[1] * pt[1];
            d_intrinsic_mat_inv(1, 2) += d_dir[1] * pt[2];
            d_intrinsic_mat_inv(2, 0) += d_dir[2] * pt[0];
            d_intrinsic_mat_inv(2, 1) += d_dir[2] * pt[1];
            d_intrinsic_mat_inv(2, 2) += d_dir[2] * pt[2];
            atomic_add(d_camera.intrinsic_mat_inv, d_intrinsic_mat_inv);
            // org = xfm_point(camera.cam_to_world, Vector3{0, 0, 0})
            auto d_cam_org = Vector3{0, 0, 0};
            d_xfm_point(camera.cam_to_world, Vector3{0, 0, 0}, d_org,
                        d_cam_to_world, d_cam_org);
            if (camera.use_look_at) {
                auto d_p = Vector3{0, 0, 0};
                auto d_l = Vector3{0, 0, 0};
                auto d_up = Vector3{0, 0, 0};
                d_look_at_matrix(camera.position, camera.look, camera.up,
                    d_cam_to_world, d_p, d_l, d_up);
                atomic_add(d_camera.position, d_p);
                atomic_add(d_camera.look, d_l);
                atomic_add(d_camera.up, d_up);
            } else {
                atomic_add(d_camera.cam_to_world, d_cam_to_world);
            }
            if (d_screen_pos != nullptr) {
                // dir = camera.intrinsic_mat_inv * pt
                auto d_pt = d_dir * camera.intrinsic_mat_inv;
                // pt = Vector3{(screen_pos[0] - 0.5f) * 2.f,
                //              (screen_pos[1] - 0.5f) * (-2.f) / aspect_ratio,
                //              Real(1)};
                (*d_screen_pos)[0] += d_pt[0] * 2;
                (*d_screen_pos)[1] += d_pt[1] * (-2 / aspect_ratio);
            }
        } break;
        case CameraType::Orthographic: {
            // Linear projection
            // [0, 1] x [0, 1] -> [-1, 1/aspect_ratio] x [1, -1/aspect_ratio]
            auto aspect_ratio = Real(camera.width) / Real(camera.height);
            auto pt = Vector3{(screen_pos[0] - 0.5f) * 2.f,
                               (screen_pos[1] - 0.5f) * (-2.f) / aspect_ratio,
                               Real(1)};
            auto local_org = camera.intrinsic_mat_inv * pt;
            // auto org = xfm_point(camera.cam_to_world, local_org);
            auto dir = xfm_vector(camera.cam_to_world, Vector3{0, 0, 1});
            // auto n_dir = normalize(dir);

            auto d_org = d_ray.org;
            auto d_n_dir = d_ray.dir;
            auto d_dir = d_normalize(dir, d_n_dir);
            auto d_local_dir = Vector3{0, 0, 0};
            auto d_cam_to_world = Matrix4x4();
            d_xfm_vector(camera.cam_to_world, Vector3{0, 0, 1}, d_dir,
                         d_cam_to_world, d_local_dir);
            auto d_local_org = Vector3{0, 0, 0};
            d_xfm_point(camera.cam_to_world, local_org, d_org,
                        d_cam_to_world, d_local_org);
            // local_org = camera.intrinsic_mat_inv * pt
            auto d_intrinsic_mat_inv = Matrix3x3{};
            d_intrinsic_mat_inv(0, 0) += d_local_org[0] * pt[0];
            d_intrinsic_mat_inv(0, 1) += d_local_org[0] * pt[1];
            d_intrinsic_mat_inv(0, 2) += d_local_org[0] * pt[2];
            d_intrinsic_mat_inv(1, 0) += d_local_org[1] * pt[0];
            d_intrinsic_mat_inv(1, 1) += d_local_org[1] * pt[1];
            d_intrinsic_mat_inv(1, 2) += d_local_org[1] * pt[2];
            d_intrinsic_mat_inv(2, 0) += d_local_org[2] * pt[0];
            d_intrinsic_mat_inv(2, 1) += d_local_org[2] * pt[1];
            d_intrinsic_mat_inv(2, 2) += d_local_org[2] * pt[2];
            atomic_add(d_camera.intrinsic_mat_inv, d_intrinsic_mat_inv);
            if (camera.use_look_at) {
                auto d_p = Vector3{0, 0, 0};
                auto d_l = Vector3{0, 0, 0};
                auto d_up = Vector3{0, 0, 0};
                d_look_at_matrix(camera.position, camera.look, camera.up,
                    d_cam_to_world, d_p, d_l, d_up);
                atomic_add(d_camera.position, d_p);
                atomic_add(d_camera.look, d_l);
                atomic_add(d_camera.up, d_up);
            } else {
                atomic_add(d_camera.cam_to_world, d_cam_to_world);
            }

            if (d_screen_pos != nullptr) {
                // local_org = camera.intrinsic_mat_inv * pt
                auto d_pt = d_local_org * camera.intrinsic_mat_inv;
                // pt = Vector3{(screen_pos[0] - 0.5f) * 2.f,
                //              (screen_pos[1] - 0.5f) * (-2.f) / aspect_ratio,
                //              Real(1)};
                (*d_screen_pos)[0] += d_pt[0] * 2;
                (*d_screen_pos)[1] += d_pt[1] * (-2 / aspect_ratio);
            }
        } break;
        case CameraType::Fisheye: {
            // Equi-angular projection
            // auto org = xfm_point(camera.cam_to_world, Vector3{0, 0, 0});
            // x, y to polar coordinate
            auto x = 2.f * (screen_pos[0] - 0.5f);
            auto y = 2.f * (screen_pos[1] - 0.5f);
            if (x * x + y * y > 1.f) {
                return;
            }
            auto r = sqrt(x*x + y*y);
            auto phi = atan2(y, x);
            // polar coordinate to spherical, map r to angle through polynomial
            auto theta = r * Real(M_PI) / 2.f;
            auto sin_phi = sin(phi);
            auto cos_phi = cos(phi);
            auto sin_theta = sin(theta);
            auto cos_theta = cos(theta);
            auto dir = Vector3{-cos_phi * sin_theta,
                               -sin_phi * sin_theta,
                               cos_theta};
            auto world_dir = xfm_vector(camera.cam_to_world, dir);
            // auto n_world_dir = normalize(world_dir);

            // ray = Ray{org, world_dir};
            auto d_org = d_ray.org;
            auto d_n_world_dir = d_ray.dir;
            auto d_world_dir = d_normalize(world_dir, d_n_world_dir);
            auto d_dir = Vector3{0, 0, 0};
            // world_dir = xfm_vector(camera.cam_to_world, dir)
            auto d_cam_to_world = Matrix4x4();
            d_xfm_vector(camera.cam_to_world, dir, d_world_dir,
                         d_cam_to_world, d_dir);
            // org = xfm_point(camera.cam_to_world, Vector3{0, 0, 0})
            auto cam_org = Vector3{0, 0, 0};
            d_xfm_point(camera.cam_to_world, Vector3{0, 0, 0}, d_org,
                        d_cam_to_world, cam_org);
            if (camera.use_look_at) {
                auto d_p = Vector3{0, 0, 0};
                auto d_l = Vector3{0, 0, 0};
                auto d_up = Vector3{0, 0, 0};
                d_look_at_matrix(camera.position, camera.look, camera.up,
                    d_cam_to_world, d_p, d_l, d_up);
                atomic_add(d_camera.position, d_p);
                atomic_add(d_camera.look, d_l);
                atomic_add(d_camera.up, d_up);
            } else {
                atomic_add(d_camera.cam_to_world, d_cam_to_world);
            }

            if (d_screen_pos != nullptr) {
                // dir = Vector3{-cos_phi * sin_theta,
                //               -sin_phi * sin_theta,
                //               cos_theta};
                auto d_cos_phi = d_dir[0] * (-sin_theta);
                auto d_sin_phi = d_dir[1] * (-sin_theta);
                auto d_sin_theta = d_dir[0] * (-cos_phi) + d_dir[1] * (-sin_phi);
                auto d_cos_theta = d_dir[2];
                // sin_phi = sin(phi)
                // cos_phi = cos(phi)
                auto d_phi = d_cos_phi * (-sin_phi) + d_sin_phi * cos_phi;
                // sin_theta = sin(theta)
                // cos_theta = cos(theta)
                auto d_theta = d_cos_theta * (-sin_theta) + d_sin_theta * cos_theta;
                // theta = r * Real(M_PI) / 2.f
                auto d_r = d_theta * (Real(M_PI) / 2.f);
                // phi = atan2(y, x)
                auto d_x = d_phi * (-y / (x * x + y * y));
                auto d_y = d_phi * (x / (x * x + y * y));
                // r = sqrt(x*x + y*y)
                d_x += (d_r * (x / r));
                d_y += (d_r * (y / r));
                // auto x = 2.f * (screen_pos[0] - 0.5f);
                // auto y = 2.f * (screen_pos[1] - 0.5f);
                (*d_screen_pos)[0] += 2 * d_x;
                (*d_screen_pos)[1] += 2 * d_y;
            }
        } break;
        case CameraType::Panorama: {
            // x, y to spherical coordinate
            auto theta = Real(M_PI) * screen_pos.y;
            auto phi = Real(2 * M_PI) * screen_pos.x;
            auto sin_phi = sin(phi);
            auto cos_phi = cos(phi);
            auto sin_theta = sin(theta);
            auto cos_theta = cos(theta);
            auto dir = Vector3{cos_phi * sin_theta,
                               cos_theta,
                               sin_phi * sin_theta};
            auto world_dir = xfm_vector(camera.cam_to_world, dir);
            // auto n_world_dir = normalize(world_dir);

            // ray = Ray{org, world_dir};
            auto d_org = d_ray.org;
            auto d_n_world_dir = d_ray.dir;
            auto d_world_dir = d_normalize(world_dir, d_n_world_dir);
            auto d_dir = Vector3{0, 0, 0};
            // world_dir = xfm_vector(camera.cam_to_world, dir)
            auto d_cam_to_world = Matrix4x4();
            d_xfm_vector(camera.cam_to_world, dir, d_world_dir,
                         d_cam_to_world, d_dir);
            // No need to propagate to x, y
            // org = xfm_point(camera.cam_to_world, Vector3{0, 0, 0})
            auto cam_org = Vector3{0, 0, 0};
            d_xfm_point(camera.cam_to_world, Vector3{0, 0, 0}, d_org,
                        d_cam_to_world, cam_org);
            if (camera.use_look_at) {
                auto d_p = Vector3{0, 0, 0};
                auto d_l = Vector3{0, 0, 0};
                auto d_up = Vector3{0, 0, 0};
                d_look_at_matrix(camera.position, camera.look, camera.up,
                    d_cam_to_world, d_p, d_l, d_up);
                atomic_add(d_camera.position, d_p);
                atomic_add(d_camera.look, d_l);
                atomic_add(d_camera.up, d_up);
            } else {
                atomic_add(d_camera.cam_to_world, d_cam_to_world);
            }

            if (d_screen_pos != nullptr) {
                // dir = Vector3{cos_phi * sin_theta,
                //               cos_theta,
                //               sin_phi * sin_theta};
                auto d_cos_phi = d_dir[0] * sin_theta;
                auto d_sin_phi = d_dir[2] * sin_theta;
                auto d_sin_theta = d_dir[0] * cos_phi + d_dir[2] * sin_phi;
                auto d_cos_theta = d_dir[1];
                // sin_phi = sin(phi)
                // cos_phi = cos(phi)
                auto d_phi = d_cos_phi * (-sin_phi) + d_sin_phi * cos_phi;
                // sin_theta = sin(theta)
                // cos_theta = cos(theta)
                auto d_theta = d_cos_theta * (-sin_theta) + d_sin_theta * cos_theta;
                // theta = Real(M_PI) * screen_pos.y
                // phi = Real(2 * M_PI) * screen_pos.x
                (*d_screen_pos)[0] += d_phi * Real(2 * M_PI);
                (*d_screen_pos)[1] += d_theta * Real(M_PI);
            }
        } break;
        default: {
            assert(false);
        }
    }
}

void sample_primary_rays(const Camera &cam,
                         const BufferView<CameraSample> &samples,
                         BufferView<Ray> rays,
                         BufferView<RayDifferential> ray_differentials,
                         bool use_gpu);

template <typename T>
DEVICE
TVector2<T> camera_to_screen(const Camera &camera,
                             const TVector3<T> &pt) {
    switch(camera.camera_type) {
        case CameraType::Perspective: {
            // Linear projection
            auto aspect_ratio = Real(camera.width) / Real(camera.height);
            auto Ipt3 = camera.intrinsic_mat * pt;
            auto Ipt = Vector2{Ipt3[0] / Ipt3[2], Ipt3[1] / Ipt3[2]};
            // [-1, 1/aspect_ratio] x [1, -1/aspect_ratio] -> [0, 1] x [0, 1]
            auto x = (Ipt[0] + 1.f) * 0.5f;
            auto y = (-Ipt[1] * aspect_ratio + 1.f) * 0.5f;
            return TVector2<T>{x, y};
        }
        case CameraType::Orthographic: {
            // Linear projection
            auto aspect_ratio = Real(camera.width) / Real(camera.height);
            auto Ipt = camera.intrinsic_mat * pt;
            // drop Ipt[2]
            // [-1, 1/aspect_ratio] x [1, -1/aspect_ratio] -> [0, 1] x [0, 1]
            auto x = (Ipt[0] + 1.f) * 0.5f;
            auto y = (-Ipt[1] * aspect_ratio + 1.f) * 0.5f;
            return TVector2<T>{x, y};
        }
        case CameraType::Fisheye: {
            // Equi-angular projection
            auto dir = normalize(pt);
            auto cos_theta = dir[2];
            auto phi = atan2(dir[1], dir[0]);
            auto theta = acos(cos_theta);
            auto r = theta * 2.f / Real(M_PI);
            auto x = 0.5f * (-r * cos(phi) + 1.f);
            auto y = 0.5f * (-r * sin(phi) + 1.f);
            return TVector2<T>{x, y};
        }
        case CameraType::Panorama: {
            // Find x, y from local dir
            auto dir = normalize(pt);
            auto cos_theta = dir[1];
            auto phi = atan2(dir[2], dir[0]);
            auto theta = acos(cos_theta);
            auto x = phi / Real(2 * M_PI);
            auto y = theta / Real(M_PI);
            return TVector2<T>{x, y};
        }
        default: {
            assert(false);
            return TVector2<T>{T(0), T(0)};
        }
    }
}

template <typename T>
DEVICE
inline void d_camera_to_screen(const Camera &camera,
                               const TVector3<T> &pt,
                               T dx, T dy,
                               DCamera &d_camera,
                               TVector3<T> &d_pt) {
    switch(camera.camera_type) {
        case CameraType::Perspective: {
            auto aspect_ratio = Real(camera.width) / Real(camera.height);
            auto Ipt3 = camera.intrinsic_mat * pt;
            auto Ipt = Vector2{Ipt3[0] / Ipt3[2], Ipt3[1] / Ipt3[2]};
            // [-1, 1/aspect_ratio] x [1, -1/aspect_ratio] -> [0, 1] x [0, 1]
            // auto x = (Ipt[0] + 1.f) * 0.5f;
            // auto y = (-Ipt[1] * aspect_ratio + 1.f) * 0.5f;

            auto d_Ipt = Vector2{dx * 0.5f, dy * -0.5f * aspect_ratio};
            // Ipt = Vector2{Ipt3[0] / Ipt3[2], Ipt3[1] / Ipt3[2]}
            auto d_Ipt3 = Vector3{d_Ipt[0] / Ipt3[2],
                                  d_Ipt[1] / Ipt3[2],
                                  - (d_Ipt[0] * Ipt[0] / Ipt3[2] +
                                     d_Ipt[1] * Ipt[1] / Ipt3[2])};
            // Ipt = camera.intrinsic_mat * pt
            auto d_intrinsic_mat = Matrix3x3{};
            d_intrinsic_mat(0, 0) += d_Ipt3[0] * pt[0];
            d_intrinsic_mat(0, 1) += d_Ipt3[0] * pt[1];
            d_intrinsic_mat(0, 2) += d_Ipt3[0] * pt[2];
            d_intrinsic_mat(1, 0) += d_Ipt3[1] * pt[0];
            d_intrinsic_mat(1, 1) += d_Ipt3[1] * pt[1];
            d_intrinsic_mat(1, 2) += d_Ipt3[1] * pt[2];
            d_intrinsic_mat(2, 0) += d_Ipt3[2] * pt[0];
            d_intrinsic_mat(2, 1) += d_Ipt3[2] * pt[1];
            d_intrinsic_mat(2, 2) += d_Ipt3[2] * pt[2];
            atomic_add(d_camera.intrinsic_mat, d_intrinsic_mat);
            d_pt[0] += d_Ipt3[0] * camera.intrinsic_mat(0, 0) +
                       d_Ipt3[1] * camera.intrinsic_mat(1, 0) +
                       d_Ipt3[2] * camera.intrinsic_mat(2, 0);
            d_pt[1] += d_Ipt3[0] * camera.intrinsic_mat(0, 1) +
                       d_Ipt3[1] * camera.intrinsic_mat(1, 1) +
                       d_Ipt3[2] * camera.intrinsic_mat(2, 1);
            d_pt[2] += d_Ipt3[0] * camera.intrinsic_mat(0, 2) +
                       d_Ipt3[1] * camera.intrinsic_mat(1, 2) +
                       d_Ipt3[2] * camera.intrinsic_mat(2, 2);
        } break;
        case CameraType::Orthographic: {
            auto aspect_ratio = Real(camera.width) / Real(camera.height);
            // auto Ipt = camera.intrinsic_mat * pt;
            // [-1, 1/aspect_ratio] x [1, -1/aspect_ratio] -> [0, 1] x [0, 1]
            // auto x = (Ipt[0] + 1.f) * 0.5f;
            // auto y = (-Ipt[1] * aspect_ratio + 1.f) * 0.5f;

            auto d_Ipt = Vector2{dx * 0.5f, dy * -0.5f * aspect_ratio};
            // Ipt = camera.intrinsic_mat * pt
            auto d_intrinsic_mat = Matrix3x3{};
            d_intrinsic_mat(0, 0) += d_Ipt[0] * pt[0];
            d_intrinsic_mat(0, 1) += d_Ipt[0] * pt[1];
            d_intrinsic_mat(0, 2) += d_Ipt[0] * pt[2];
            d_intrinsic_mat(1, 0) += d_Ipt[1] * pt[0];
            d_intrinsic_mat(1, 1) += d_Ipt[1] * pt[1];
            d_intrinsic_mat(1, 2) += d_Ipt[1] * pt[2];
            atomic_add(d_camera.intrinsic_mat, d_intrinsic_mat);
            d_pt[0] += d_Ipt[0] * camera.intrinsic_mat(0, 0) +
                       d_Ipt[1] * camera.intrinsic_mat(1, 0);
            d_pt[1] += d_Ipt[0] * camera.intrinsic_mat(0, 1) +
                       d_Ipt[1] * camera.intrinsic_mat(1, 1);
            d_pt[2] += d_Ipt[0] * camera.intrinsic_mat(0, 2) +
                       d_Ipt[1] * camera.intrinsic_mat(1, 2);
        } break;
        case CameraType::Fisheye: {
            auto dir = normalize(pt);
            auto phi = atan2(dir[1], dir[0]);
            auto theta = acos(dir[2]);
            auto r = theta * 2.f / Real(M_PI);
            // x = 0.5f * (-r * cos(phi) + 1.f)
            // y = 0.5f * (-r * sin(phi) + 1.f)
            auto dr = -0.5f * (cos(phi) * dx + sin(phi) * dy);
            auto dphi = 0.5f * r * sin(phi) * dx -
                        0.5f * r * cos(phi) * dy;
            // r = theta * 2.f / float(M_PI)
            auto dtheta = dr * (2.f / Real(M_PI));
            // theta = acos(cos_theta)
            auto d_cos_theta = -dtheta / sqrt(1.f - dir[2] * dir[2]);
            // phi = atan2(dir[1], dir[0])
            auto atan2_tmp = dir[0] * dir[0] + dir[1] * dir[1];
            auto ddir0 = -dphi * dir[1] / atan2_tmp;
            auto ddir1 =  dphi * dir[0] / atan2_tmp;
            // cos_theta = dir[2]
            auto ddir2 = d_cos_theta;
            // Backprop dir = normalize(pt);
            auto ddir = Vector3{ddir0, ddir1, ddir2};
            d_pt += d_normalize(pt, ddir);
        } break;
        case CameraType::Panorama: {
            // Find x, y from local dir
            auto dir = normalize(pt);
            // auto cos_theta = dir[1];
            // auto phi = atan2(dir[2], dir[0]);
            // auto theta = acos(cos_theta);
            // auto x = phi / Real(2 * M_PI);
            // auto y = theta / Real(M_PI);
            auto d_phi = dx / Real(2 * M_PI);
            auto d_theta = dy / Real(M_PI);
            // theta = acos(cos_theta)
            auto d_cos_theta = -d_theta / sqrt(1.f - dir[1] * dir[1]);
            // phi = atan2(dir[2], dir[0])
            auto atan2_tmp = dir[0] * dir[0] + dir[2] * dir[2];
            auto ddir0 = -d_phi * dir[2] / atan2_tmp;
            auto ddir2 =  d_phi * dir[0] / atan2_tmp;
            // cos_theta = dir[1]
            auto ddir1 = d_cos_theta;
            // Backprop dir = normalize(pt);
            auto ddir = Vector3{ddir0, ddir1, ddir2};
            d_pt += d_normalize(pt, ddir);
        } break;
        default: {
            assert(false);
        }
    }
}

template <typename T>
DEVICE
bool project(const Camera &camera,
             const TVector3<T> &p0,
             const TVector3<T> &p1,
             TVector2<T> &pp0,
             TVector2<T> &pp1) {
    auto p0_local = xfm_point(camera.world_to_cam, p0);
    auto p1_local = xfm_point(camera.world_to_cam, p1);
    if (p0_local[2] < camera.clip_near && p1_local[2] < camera.clip_near) {
        return false;
    }
    // clip against z = clip_near
    if (p0_local[2] < camera.clip_near) {
        // a ray from p1 to p0
        auto dir = p0_local - p1_local;
        // intersect with plane z = clip_near
        auto t = -(p1_local[2] - camera.clip_near) / dir[2];
        p0_local = p1_local + t * dir;
    } else if (p1_local[2] < camera.clip_near) {
        // a ray from p1 to p0
        auto dir = p1_local - p0_local;
        // intersect with plane z = clip_near
        auto t = -(p0_local[2] - camera.clip_near) / dir[2];
        p1_local = p0_local + t * dir;
    }
    // project to 2d screen
    pp0 = camera_to_screen(camera, p0_local);
    pp1 = camera_to_screen(camera, p1_local);
    return true;
}

DEVICE
inline void d_project(const Camera &camera,
                      const Vector3 &p0,
                      const Vector3 &p1,
                      Real dp0x, Real dp0y,
                      Real dp1x, Real dp1y,
                      DCamera &d_camera,
                      Vector3 &d_p0,
                      Vector3 &d_p1) {
    auto p0_local = xfm_point(camera.world_to_cam, p0);
    auto p1_local = xfm_point(camera.world_to_cam, p1);
    if (p0_local[2] < camera.clip_near && p1_local[2] < camera.clip_near) {
        return;
    }
    auto clipped_p0_local = p0_local;
    auto clipped_p1_local = p1_local;
    // clip against z = clip_near
    if (p0_local[2] < camera.clip_near) {
        // a ray from p1 to p0
        auto dir = p0_local - p1_local;
        // intersect with plane z = clip_near
        auto t = -(p1_local[2] - camera.clip_near) / dir[2];
        clipped_p0_local = p1_local + t * dir;
    } else if (p1_local[2] < camera.clip_near) {
        // a ray from p1 to p0
        auto dir = p1_local - p0_local;
        // intersect with plane z = clip_near
        auto t = -(p0_local[2] - camera.clip_near) / dir[2];
        clipped_p1_local = p0_local + t * dir;
    }

    // p0' = camera_to_screen(camera, clipped_p0_local)
    // p1' = camera_to_screen(camera, clipped_p1_local)
    auto dclipped_p0_local = Vector3{0, 0, 0};
    auto dclipped_p1_local = Vector3{0, 0, 0};
    d_camera_to_screen(camera, clipped_p0_local,
        dp0x, dp0y, d_camera, dclipped_p0_local);
    d_camera_to_screen(camera, clipped_p1_local,
        dp1x, dp1y, d_camera, dclipped_p1_local);

    auto dp0_local = Vector3{0.f, 0.f, 0.f};
    auto dp1_local = Vector3{0.f, 0.f, 0.f};
    // differentiate through clipping
    if (p0_local[2] < camera.clip_near) {
        auto dir = p0_local - p1_local;
        auto t = -(p1_local[2] + camera.clip_near) / dir[2];
        // clipped_p0_local = p1_local + t * dir
        dp1_local += dclipped_p0_local;
        auto dt = dot(dir, dclipped_p0_local);
        auto ddir = t * dclipped_p0_local;
        // t = -p1_local[2] / dir[2]
        dp1_local[2] += (-dt / dir[2]);
        ddir[2] -= dt * t / dir[2];
        // dir = p0_local - p1_local;
        dp0_local += ddir;
        dp1_local -= ddir;
        // clipped_p1_local = p1_local
        dp1_local += dclipped_p1_local;
    } else if (p1_local[2] < camera.clip_near) {
        auto dir = p1_local - p0_local;
        auto t = -(p0_local[2] + camera.clip_near) / dir[2];
        // clipped_p1_local = p0_local + t * dir
        dp0_local += dclipped_p1_local;
        auto dt = dot(dir, dclipped_p1_local);
        auto ddir = t * dclipped_p1_local;
        // t = -p0_local[2] / dir[2]
        dp0_local[2] += (-dt / dir[2]);
        ddir[2] -= dt * t / dir[2];
        // dir = p1_local - p0_local;
        dp1_local += ddir;
        dp0_local -= ddir;
        // clipped_p0_local = p0_local
        dp0_local += dclipped_p0_local;
    } else {
        dp0_local += dclipped_p0_local;
        dp1_local += dclipped_p1_local;
    }

    // p0_local = xfm_point(camera.world_to_cam, p0)
    // p1_local = xfm_point(camera.world_to_cam, p1)
    auto d_world_to_cam = Matrix4x4();
    d_xfm_point(camera.world_to_cam, p0, dp0_local, d_world_to_cam, d_p0);
    d_xfm_point(camera.world_to_cam, p1, dp1_local, d_world_to_cam, d_p1);
    // http://jack.valmadre.net/notes/2016/09/04/back-prop-differentials/#back-propagation-using-differentials
    // Super cool article btw
    auto tw2c = transpose(camera.world_to_cam);
    auto d_cam_to_world = -tw2c * d_world_to_cam * tw2c;
    if (camera.use_look_at) {
        auto d_p = Vector3{0, 0, 0};
        auto d_l = Vector3{0, 0, 0};
        auto d_up = Vector3{0, 0, 0};
        d_look_at_matrix(camera.position, camera.look, camera.up,
            d_cam_to_world, d_p, d_l, d_up);
        atomic_add(d_camera.position, d_p);
        atomic_add(d_camera.look, d_l);
        atomic_add(d_camera.up, d_up);
    } else {
        atomic_add(d_camera.cam_to_world, d_cam_to_world);
    }
}

template <typename T>
DEVICE
inline TVector3<T> screen_to_camera(const Camera &camera,
                                    const TVector2<T> &screen_pos) {
    // XXX: also return position
    switch(camera.camera_type) {
        case CameraType::Perspective: {
            // Linear projection
            // [0, 1] x [0, 1] -> [1, -1] -> [1, -1]/aspect_ratio
            auto aspect_ratio = Real(camera.width) / Real(camera.height);
            auto pt = TVector3<T>{
                (screen_pos[0] - 0.5f) * 2.f,
                (screen_pos[1] - 0.5f) * -2.f / aspect_ratio,
                T(1)};
            auto dir = camera.intrinsic_mat_inv * pt;
            auto dir_n = TVector3<T>{dir[0] / dir[2], dir[1] / dir[2], T(1)};
            return dir_n;
        }
        case CameraType::Orthographic: {
            assert(false); // TODO
            return TVector3<T>{0, 0, 1};
        }
        case CameraType::Fisheye: {
            // x, y to polar coordinate
            auto x = 2.f * (screen_pos[0] - 0.5f);
            auto y = 2.f * (screen_pos[1] - 0.5f);
            auto r = sqrt(x*x + y*y);
            auto phi = atan2(y, x);
            // polar coordinate to spherical, map r linearly on angle
            auto theta = r * Real(M_PI) / 2.f;
            auto sin_phi = sin(phi);
            auto cos_phi = cos(phi);
            auto sin_theta = sin(theta);
            auto cos_theta = cos(theta);
            auto dir = TVector3<T>{
                -cos_phi * sin_theta, -sin_phi * sin_theta, cos_theta};
            return dir;
        }
        case CameraType::Panorama: {
            // x, y to polar coordinate
            auto x = screen_pos[0];
            auto y = screen_pos[1];
            auto theta = Real(M_PI) * y;
            auto phi = Real(2 * M_PI) * x;
            auto sin_phi = sin(phi);
            auto cos_phi = cos(phi);
            auto sin_theta = sin(theta);
            auto cos_theta = cos(theta);
            auto dir = TVector3<T>{cos_phi * sin_theta,
                                   cos_theta,
                                   sin_phi * sin_theta};
            return dir;
        }
        default: {
            assert(false);
            return TVector3<T>{0, 0, 0};
        }
    }
}

template <typename T>
DEVICE
inline void d_screen_to_camera(const Camera &camera,
                               const TVector2<T> &screen_pos,
                               TVector3<T> &d_x,
                               TVector3<T> &d_y) {
    switch(camera.camera_type) {
        case CameraType::Perspective: {
            auto aspect_ratio = Real(camera.width) / Real(camera.height);
            auto pt = TVector3<T>{
                (screen_pos[0] - 0.5f) * 2.f,
                (screen_pos[1] - 0.5f) * -2.f / aspect_ratio,
                T(1)};
            auto d_pt_d_x = TVector3<T>{
                2.f * screen_pos[0], T(0), T(0)};
            auto d_pt_d_y = TVector3<T>{
                T(0), -2.f * screen_pos[1] / aspect_ratio, T(0)};
            auto dir = camera.intrinsic_mat_inv * pt;
            auto d_dir_dx = camera.intrinsic_mat_inv * d_pt_d_x;
            auto d_dir_dy = camera.intrinsic_mat_inv * d_pt_d_y;
            // auto dir_n = TVector3<T>{dir[0] / dir[2], dir[1] / dir[2], T(1)};
            d_x = TVector3<T>{dir[2] * d_dir_dx[0] - d_dir_dx[2] * dir[0],
                              dir[2] * d_dir_dx[1] - d_dir_dx[2] * dir[1],
                              T(0)} / square(dir[2]);
            d_y = TVector3<T>{dir[2] * d_dir_dy[0] - d_dir_dy[2] * dir[0],
                              dir[2] * d_dir_dy[1] - d_dir_dy[2] * dir[1],
                              T(0)} / square(dir[2]);
        } break;
        case CameraType::Orthographic: {
            assert(false); // TODO
        } break;
        case CameraType::Fisheye: {
            // x, y to polar coordinate
            auto x = 2.f * (screen_pos[0] - 0.5f);
            auto y = 2.f * (screen_pos[1] - 0.5f);
            auto r = sqrt(x*x + y*y);
            auto phi = atan2(y, x);
            // polar coordinate to spherical, map r linearly on angle
            auto theta = r * Real(M_PI) / 2.f;
            auto sin_phi = sin(phi);
            auto cos_phi = cos(phi);
            auto sin_theta = sin(theta);
            auto cos_theta = cos(theta);
     
            // d dir d screen_pos:
            // auto dir = TVector3<T>{
            //     -cos_phi * sin_theta, -sin_phi * sin_theta, cos_theta};
            auto d_dir_x_d_phi = sin_phi * sin_theta;
            auto d_dir_x_d_theta = -cos_phi * cos_theta;
            auto d_dir_y_d_phi = -cos_phi * sin_theta;
            auto d_dir_y_d_theta = -sin_phi * cos_theta;
            auto d_dir_z_d_theta = -sin_theta;
            auto d_phi_d_x = -y / (r*r);
            auto d_phi_d_y = x / (r*r);
            auto d_theta_d_x = (float(M_PI) / 2.f) * x / r;
            auto d_theta_d_y = (float(M_PI) / 2.f) * y / r;

            d_x = 2.f * TVector3<T>{
                d_dir_x_d_phi * d_phi_d_x + d_dir_x_d_theta * d_theta_d_x,
                d_dir_y_d_phi * d_phi_d_x + d_dir_y_d_theta * d_theta_d_x,
                d_dir_z_d_theta * d_theta_d_x};
            d_y = 2.f * TVector3<T>{
                d_dir_x_d_phi * d_phi_d_y + d_dir_x_d_theta * d_theta_d_y,
                d_dir_y_d_phi * d_phi_d_y + d_dir_y_d_theta * d_theta_d_y,
                d_dir_z_d_theta * d_theta_d_y};
        } break;
        case CameraType::Panorama: {
            auto x = screen_pos[0];
            auto y = screen_pos[1];
            auto theta = Real(M_PI) * y;
            auto phi = Real(2 * M_PI) * x;
            auto sin_phi = sin(phi);
            auto cos_phi = cos(phi);
            auto sin_theta = sin(theta);
            auto cos_theta = cos(theta);
     
            // d dir d screen_pos:
            // auto dir = TVector3<T>{
            //     cos_phi * sin_theta, cos_theta, sin_phi * sin_theta};
            auto d_dir_x_d_phi = -sin_phi * sin_theta;
            auto d_dir_x_d_theta = cos_phi * cos_theta;
            auto d_dir_y_d_theta = -sin_theta;
            auto d_dir_z_d_phi = cos_phi * sin_theta;
            auto d_dir_z_d_theta = sin_phi * cos_theta;
            auto d_phi_d_x = Real(2 * M_PI);
            auto d_phi_d_y = Real(0);
            auto d_theta_d_x = Real(0);
            auto d_theta_d_y = Real(M_PI);

            d_x = TVector3<T>{
                d_dir_x_d_phi * d_phi_d_x + d_dir_x_d_theta * d_theta_d_x,
                d_dir_y_d_theta * d_theta_d_x,
                d_dir_z_d_phi * d_phi_d_x + d_dir_z_d_theta * d_theta_d_x};
            d_y = TVector3<T>{
                d_dir_x_d_phi * d_phi_d_y + d_dir_x_d_theta * d_theta_d_y,
                d_dir_z_d_theta * d_theta_d_y,
                d_dir_z_d_phi * d_phi_d_y + d_dir_z_d_theta * d_theta_d_y};
        } break;
        default: {
            assert(false);
        }
    }
}

DEVICE
inline bool in_screen(const Camera &cam, const Vector2 &pt) {
    if (cam.camera_type != CameraType::Fisheye) {
        return pt[0] >= 0.f && pt[0] < 1.f &&
               pt[1] >= 0.f && pt[1] < 1.f;
    } else {
        auto dist_sq =
            (pt[0] - 0.5f) * (pt[0] - 0.5f) + (pt[1] - 0.5f) * (pt[1] - 0.5f);
        return dist_sq < 0.25f;
    }
}

void test_sample_primary_rays(bool use_gpu);
void test_camera_derivatives();

#include "primary_intersection.h"
#include "scene.h"
#include "parallel.h"

struct d_primary_intersector {
    DEVICE void operator()(int idx) {
        // Initialize derivatives
        auto pixel_idx = active_pixels[idx];
        const auto &isect = isects[pixel_idx];
        auto d_primary_ray_differential = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}
        };
        auto d_ray = d_rays[pixel_idx];

        if (isect.valid()) {
            auto shape_id = isect.shape_id;
            auto tri_id = isect.tri_id;
            const auto &shape = shapes[shape_id];
            auto ind = get_indices(shape, tri_id);
            Vector3 d_v_p[3] = {
                Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            Vector3 d_v_n[3] = {
                Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            Vector2 d_v_uv[3] = {
                Vector2{0, 0}, Vector2{0, 0}, Vector2{0, 0}};
            d_intersect_shape(shape,
                              tri_id,
                              rays[pixel_idx],
                              primary_ray_differentials[pixel_idx],
                              d_points[pixel_idx],
                              d_ray_differentials[pixel_idx],
                              d_ray,
                              d_primary_ray_differential,
                              d_v_p,
                              d_v_n,
                              d_v_uv);
            atomic_add(&d_shapes[shape_id].vertices[3 * ind[0]], d_v_p[0]);
            atomic_add(&d_shapes[shape_id].vertices[3 * ind[1]], d_v_p[1]);
            atomic_add(&d_shapes[shape_id].vertices[3 * ind[2]], d_v_p[2]);
            if (has_uvs(shape)) {
                atomic_add(&d_shapes[shape_id].uvs[2 * ind[0]], d_v_uv[0]);
                atomic_add(&d_shapes[shape_id].uvs[2 * ind[1]], d_v_uv[1]);
                atomic_add(&d_shapes[shape_id].uvs[2 * ind[2]], d_v_uv[2]);
            }
            if (has_shading_normals(shape)) {
                atomic_add(&d_shapes[shape_id].normals[3 * ind[0]], d_v_n[0]);
                atomic_add(&d_shapes[shape_id].normals[3 * ind[1]], d_v_n[1]);
                atomic_add(&d_shapes[shape_id].normals[3 * ind[2]], d_v_n[2]);
            }
        }

        auto pixel_x = pixel_idx % camera.width;
        auto pixel_y = pixel_idx / camera.width;
        auto sample = samples[pixel_idx].xy;
        auto screen_pos = Vector2{
            (pixel_x + sample[0]) / Real(camera.width),
            (pixel_y + sample[1]) / Real(camera.height)
        };

        // Ray differential computation
        auto delta = Real(1e-3);
        auto screen_pos_dx = screen_pos + Vector2{delta, Real(0)};
        // auto ray_dx = sample_primary(camera, screen_pos_dx);
        auto screen_pos_dy = screen_pos + Vector2{Real(0), delta};
        // auto ray_dy = sample_primary(camera, screen_pos_dx);
        auto pixel_size_x = Real(0.5) / camera.width;
        auto pixel_size_y = Real(0.5) / camera.height;
        // auto org_dx = pixel_size_x * (ray_dx.org - ray.org) / delta;
        // auto org_dy = pixel_size_y * (ray_dy.org - ray.org) / delta;
        // auto dir_dx = pixel_size_x * (ray_dx.dir - ray.dir) / delta;
        // auto dir_dy = pixel_size_y * (ray_dy.dir - ray.dir) / delta;
        // ray_differentials[idx] = RayDifferential{org_dx, org_dy, dir_dx, dir_dy}
        auto d_ray_dx = DRay{d_primary_ray_differential.org_dx * pixel_size_x / delta,
                             d_primary_ray_differential.dir_dx * pixel_size_x / delta};
        auto d_ray_dy = DRay{d_primary_ray_differential.org_dy * pixel_size_y / delta,
                             d_primary_ray_differential.dir_dy * pixel_size_y / delta};
        d_ray.org += (d_primary_ray_differential.org_dx * -pixel_size_x +
                      d_primary_ray_differential.org_dy * -pixel_size_y) / delta;
        d_ray.dir += (d_primary_ray_differential.dir_dx * -pixel_size_x +
                      d_primary_ray_differential.dir_dy * -pixel_size_y) / delta;

        d_sample_primary_ray(camera, screen_pos, d_ray, d_camera);
        d_sample_primary_ray(camera, screen_pos_dx, d_ray_dx, d_camera);
        d_sample_primary_ray(camera, screen_pos_dy, d_ray_dy, d_camera);
    }

    const Camera camera;
    const Shape *shapes;
    const int *active_pixels;
    const CameraSample *samples;
    const Ray *rays;
    const RayDifferential *primary_ray_differentials;
    const Intersection *isects;
    const DRay *d_rays;
    const RayDifferential *d_ray_differentials;
    const SurfacePoint *d_points;
    DShape *d_shapes;
    DCamera d_camera;
};

void d_primary_intersection(const Scene &scene,
                            const BufferView<int> &active_pixels,
                            const BufferView<CameraSample> &samples,
                            const BufferView<Ray> &rays,
                            const BufferView<RayDifferential> &primary_ray_differentials,
                            const BufferView<Intersection> &intersections,
                            const BufferView<DRay> &d_rays,
                            const BufferView<RayDifferential> &d_ray_differentials,
                            const BufferView<SurfacePoint> &d_surface_points,
                            DScene *d_scene) {
    parallel_for(d_primary_intersector{
        scene.camera,
        scene.shapes.data,
        active_pixels.begin(),
        samples.begin(),
        rays.begin(),
        primary_ray_differentials.begin(),
        intersections.begin(),
        d_rays.begin(),
        d_ray_differentials.begin(),
        d_surface_points.begin(),
        d_scene->shapes.data,
        d_scene->camera}, active_pixels.size(), scene.use_gpu);
}

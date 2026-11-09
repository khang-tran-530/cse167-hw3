
import hw3_common;

// ---------------------------------------------------
// PART 1: Ray-sphere intersection
// ---------------------------------------------------

bool intersect_sphere(
    float3 ray_origin,
    float3 ray_dir,
    Sphere sphere,
    float t_min,
    float t_max,
    out float hit_t
) {
    float3 oc = ray_origin - sphere.center;

    float a = dot(ray_dir, ray_dir);
    float b = 2.0 * dot(oc, ray_dir);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;

    float2 roots;
    hit_t = t_max;

    if (!solve_quadratic(a, b, c, roots)) {
        return false;
    }

    float t0 = min(roots.x, roots.y);
    float t1 = max(roots.x, roots.y);

    if (t0 > t_min && t0 < t_max) {
        hit_t = t0;
        return true;
    }

    if (t1 > t_min && t1 < t_max) {
        hit_t = t1;
        return true;
    }

    return false;
}

// ---------------------------------------------------
// PART 2: Find nearest intersection in the scene
// ---------------------------------------------------

bool trace_scene(
    float3 ray_origin,
    float3 ray_dir,
    float t_min,
    float t_max,
    StructuredBuffer<Shape> shapes,
    int num_shapes,
    StructuredBuffer<Sphere> spheres,
    StructuredBuffer<TriangleMesh> meshes,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<float3> vertex_normals,
    StructuredBuffer<uint3> triangles,
    out float hit_t,
    out int hit_shape,
    out float3 hit_normal
) {
    hit_t = t_max;
    hit_shape = -1;
    hit_normal = float3(0.0, 0.0, 0.0);

    for (int i = 0; i < num_shapes; i++) {
        Shape shape = shapes[i];

        // Transform world-space ray to object space.
        float3 local_origin = xform_pt(
            shape.world_to_obj, ray_origin
        );

        float3 local_dir = xform_vec(
            shape.world_to_obj, ray_dir
        );

        // Do not normalize local_dir.
        float t;
        float3 local_normal;
        bool found = false;

        if (int(shape.type) == 0) {
            // -------------------------------
            // SPHERE
            // -------------------------------
            Sphere sphere = spheres[shape.index];

            if (intersect_sphere(
                local_origin,
                local_dir,
                sphere,
                t_min,
                hit_t,
                t
            )) {
                float3 local_hit =
                    local_origin + t * local_dir;

                local_normal = normalize(
                    local_hit - sphere.center
                );

                found = true;
            }
        }
        else if (int(shape.type) == 1) {
            // -------------------------------
            // TRIANGLE MESH
            // -------------------------------
            TriangleMesh mesh = meshes[shape.index];

            for (uint j = mesh.triangle_begin;
                 j < mesh.triangle_end; j++) {

                // Convert local vertex indices
                // to global buffer indices.
                uint3 tri =
                    triangles[j] + mesh.vertex_offset;

                float3 p0 = vertices[tri.x];
                float3 p1 = vertices[tri.y];
                float3 p2 = vertices[tri.z];

                float3 barycentric;
                float triangle_t;

                if (intersect_tri(
                    p0, p1, p2,
                    local_origin,
                    local_dir,
                    t_min,
                    hit_t,
                    barycentric,
                    triangle_t
                )) {
                    // Keep the closest triangle.
                    t = triangle_t;

                    float3 n0 = vertex_normals[tri.x];
                    float3 n1 = vertex_normals[tri.y];
                    float3 n2 = vertex_normals[tri.z];

                    // Interpolate vertex normals.
                    local_normal = normalize(
                        barycentric.x * n0 +
                        barycentric.y * n1 +
                        barycentric.z * n2
                    );

                    // Update closest hit immediately.
                    hit_t = t;
                    hit_shape = i;

                    hit_normal = normalize(
                        xform_vec(
                            transpose(shape.world_to_obj),
                            local_normal
                        )
                    );
                }
            }

            // The triangle branch already updated
            // the closest hit, so move to next shape.
            continue;
        }

        // Update closest hit for spheres.
        if (found && t < hit_t) {
            hit_t = t;
            hit_shape = i;

            hit_normal = normalize(
                xform_vec(
                    transpose(shape.world_to_obj),
                    local_normal
                )
            );
        }
    }

    return hit_shape >= 0;
}

// ---------------------------------------------------
// PART 3: Main ray tracing shader
// ---------------------------------------------------

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(
    uint2 tid : SV_DispatchThreadID,
    uniform float3 background,
    uniform Camera camera,
    StructuredBuffer<Shape> shapes,
    uniform int num_shapes,
    StructuredBuffer<Sphere> spheres,
    StructuredBuffer<TriangleMesh> meshes,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<float3> vertex_normals,
    StructuredBuffer<uint3> triangles,
    StructuredBuffer<PointLight> lights,
    uniform int num_lights,
    uniform int max_depth,
    RWTexture2D<float4> image
) {
    uint width, height;
    image.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // STEP 1: Generate camera ray.
    float aspect = float(width) / float(height);

    float half_width =
        tan(radians(camera.fov_x) * 0.5);

    float half_height = half_width / aspect;

    float x = (
        2.0 * (float(tid.x) + 0.5) /
        float(width) - 1.0
    ) * half_width;

    float y = (
        1.0 - 2.0 * (float(tid.y) + 0.5) /
        float(height)
    ) * half_height;

    float3 camera_dir = float3(x, y, -1.0);

    // Transform camera ray to world space.
    float3 ray_origin = xform_pt(
        camera.cam_to_world,
        float3(0.0, 0.0, 0.0)
    );

    float3 ray_dir = normalize(
        xform_vec(camera.cam_to_world, camera_dir)
    );

    // STEP 2: Initialize color and throughput.
    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);

    // STEP 3: Trace multiple reflection bounces.
    for (int bounce = 0; bounce < max_depth; bounce++) {

        float closest_t;
        int closest_shape;
        float3 normal;

        bool found = trace_scene(
            ray_origin,
            ray_dir,
            EPSILON,
            1e30,
            shapes,
            num_shapes,
            spheres,
            meshes,
            vertices,
            vertex_normals,
            triangles,
            closest_t,
            closest_shape,
            normal
        );

        // If ray misses everything, add background.
        if (!found) {
            color += throughput * background;
            break;
        }

        Shape hit_shape = shapes[closest_shape];

        float3 hit_point =
            ray_origin + closest_t * ray_dir;

        // STEP 4: Calculate diffuse lighting.
        float3 contribution = float3(0.0, 0.0, 0.0);

        for (int l = 0; l < num_lights; l++) {
            PointLight light = lights[l];

            float3 to_light =
                light.position - hit_point;

            float distance_sq =
                dot(to_light, to_light);

            float distance = sqrt(distance_sq);

            if (distance <= EPSILON) {
                continue;
            }

            float3 light_dir = to_light / distance;

            float NdotL = max(
                0.0, dot(normal, light_dir)
            );

            if (NdotL <= 0.0) {
                continue;
            }

            // STEP 5: Shadow ray.
            float shadow_t;
            int shadow_shape;
            float3 shadow_normal;

            bool in_shadow = trace_scene(
                hit_point,
                light_dir,
                EPSILON,
                distance,
                shapes,
                num_shapes,
                spheres,
                meshes,
                vertices,
                vertex_normals,
                triangles,
                shadow_t,
                shadow_shape,
                shadow_normal
            );

            if (!in_shadow) {
                contribution +=
                    hit_shape.Kd *
                    light.intensity *
                    (NdotL / distance_sq);
            }
        }

        // STEP 6: Accumulate lighting.
        color += throughput * contribution;

        // STEP 7: Update reflection throughput.
        throughput *= hit_shape.Kr;

        if (all(throughput == float3(0.0, 0.0, 0.0))) {
            break;
        }

        // STEP 8: Generate reflected ray.
        ray_origin = hit_point;

        ray_dir = normalize(
            reflect(ray_dir, normal)
        );
    }

    // STEP 9: Save pixel.
    image[tid] = float4(color, 1.0);
}

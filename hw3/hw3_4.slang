
import hw3_common;

// Intersect a world-space ray with a transformed sphere.
bool intersect_sphere_world(
    float3 ray_origin,
    float3 ray_dir,
    Shape shape,
    Sphere sphere,
    float t_min,
    float t_max,
    out float hit_t,
    out float3 hit_normal
) {
    // Transform the ray into the sphere's object space.
    float3 local_origin =
        xform_pt(shape.world_to_obj, ray_origin);

    float3 local_dir =
        xform_vec(shape.world_to_obj, ray_dir);

    // Do NOT normalize local_dir.
    float3 oc = local_origin - sphere.center;

    float a = dot(local_dir, local_dir);
    float b = 2.0 * dot(oc, local_dir);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;

    float2 roots;
    hit_t = t_max;
    hit_normal = float3(0.0, 0.0, 0.0);

    if (!solve_quadratic(a, b, c, roots)) {
        return false;
    }

    float t0 = min(roots.x, roots.y);
    float t1 = max(roots.x, roots.y);

    float t = t_max;

    if (t0 > t_min && t0 < t_max) {
        t = t0;
    }
    else if (t1 > t_min && t1 < t_max) {
        t = t1;
    }
    else {
        return false;
    }

    hit_t = t;

    float3 local_hit = local_origin + t * local_dir;

    float3 local_normal = normalize(
        local_hit - sphere.center
    );

    // Inverse-transpose normal transformation.
    float4x4 normal_matrix =
        transpose(shape.world_to_obj);

    hit_normal = normalize(
        xform_vec(normal_matrix, local_normal)
    );

    return true;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(
    uint2 tid : SV_DispatchThreadID,
    uniform float3 background,
    uniform Camera camera,
    StructuredBuffer<Shape> shapes,
    uniform int num_shapes,
    StructuredBuffer<Sphere> spheres,
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

    // STEP 1: Generate ray using horizontal field of view.
    float aspect = float(width) / float(height);

    float fov = radians(camera.fov_x);
    float half_width = tan(fov * 0.5);
    float half_height = half_width / aspect;

    float x = (
        2.0 * (float(tid.x) + 0.5) / float(width) - 1.0
    ) * half_width;

    float y = (
        1.0 - 2.0 * (float(tid.y) + 0.5) / float(height)
    ) * half_height;

    float3 camera_dir = normalize(float3(x, y, -1.0));

    // Transform camera ray into world space.
    float3 ray_origin = xform_pt(
        camera.cam_to_world,
        float3(0.0, 0.0, 0.0)
    );

    float3 ray_dir = normalize(xform_vec(
        camera.cam_to_world,
        camera_dir
    ));

    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);

    // STEP 2: Trace multiple ray bounces.
    for (int bounce = 0; bounce < max_depth; bounce++) {

        float closest_t = 1e30;
        int closest_shape = -1;
        float3 closest_normal = float3(0.0, 0.0, 0.0);

        // Find the closest sphere.
        for (int i = 0; i < num_shapes; i++) {
            Shape shape = shapes[i];
            Sphere sphere = spheres[shape.index];

            float t;
            float3 normal;

            if (intersect_sphere_world(
                ray_origin,
                ray_dir,
                shape,
                sphere,
                EPSILON,
                closest_t,
                t,
                normal
            )) {
                closest_t = t;
                closest_shape = i;
                closest_normal = normal;
            }
        }

        // No intersection: use background.
        if (closest_shape == -1) {
            color += throughput * background;
            break;
        }

        Shape hit_shape = shapes[closest_shape];

        float3 hit_point =
            ray_origin + closest_t * ray_dir;

        float3 normal = closest_normal;

        // STEP 3: Diffuse lighting.
        float3 contribution = float3(0.0, 0.0, 0.0);

        for (int l = 0; l < num_lights; l++) {
            PointLight light = lights[l];

            float3 to_light = light.position - hit_point;
            float distance_sq = dot(to_light, to_light);
            float distance = sqrt(distance_sq);

            if (distance <= EPSILON) {
                continue;
            }

            float3 light_dir = to_light / distance;

            float NdotL = max(0.0, dot(normal, light_dir));

            if (NdotL <= 0.0) {
                continue;
            }

            // STEP 4: Check shadows.
            bool in_shadow = false;

            for (int j = 0; j < num_shapes; j++) {
                Shape blocker_shape = shapes[j];

                Sphere blocker =
                    spheres[blocker_shape.index];

                float shadow_t;
                float3 shadow_normal;

                if (intersect_sphere_world(
                    hit_point,
                    light_dir,
                    blocker_shape,
                    blocker,
                    EPSILON,
                    distance,
                    shadow_t,
                    shadow_normal
                )) {
                    in_shadow = true;
                    break;
                }
            }

            if (!in_shadow) {
                contribution += hit_shape.Kd *
                    light.intensity *
                    (NdotL / distance_sq);
            }
        }

        // STEP 5: Accumulate lighting.
        color += throughput * contribution;

        // STEP 6: Mirror reflection.
        throughput *= hit_shape.Kr;

        if (all(throughput == float3(0.0, 0.0, 0.0))) {
            break;
        }

        ray_origin = hit_point;
        ray_dir = normalize(reflect(ray_dir, normal));
    }

    image[tid] = float4(color, 1.0);
}

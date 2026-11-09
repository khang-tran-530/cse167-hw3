
import hw3_common;

// Find the nearest valid ray-sphere intersection.
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

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(
    uint2 tid : SV_DispatchThreadID,
    uniform float3 background,
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

    // STEP 1: Generate the primary camera ray.
    float aspect = float(width) / float(height);

    float x = (2.0 * (float(tid.x) + 0.5)
              / float(width) - 1.0) * aspect;

    float y = 1.0 - 2.0 * (float(tid.y) + 0.5)
              / float(height);

    float3 ray_origin = float3(0.0, 0.0, 0.0);
    float3 ray_dir = float3(x, y, -1.0);

    // STEP 2: Initialize accumulated color and throughput.
    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);

    // STEP 3: Trace multiple reflection bounces.
    for (int bounce = 0; bounce < max_depth; bounce++) {

        float closest_t = 1e30;
        int closest_shape = -1;

        // Find the closest sphere for the current ray.
        for (int i = 0; i < num_shapes; i++) {
            Sphere sphere = spheres[shapes[i].index];

            float t;

            if (intersect_sphere(
                ray_origin, ray_dir, sphere,
                EPSILON, closest_t, t
            )) {
                closest_t = t;
                closest_shape = i;
            }
        }

        // Ray misses everything: add background and stop.
        if (closest_shape == -1) {
            color += throughput * background;
            break;
        }

        // STEP 4: Compute the hit position and normal.
        Shape hit_shape = shapes[closest_shape];
        Sphere hit_sphere = spheres[hit_shape.index];

        float3 hit_point = ray_origin + closest_t * ray_dir;

        float3 normal = normalize(
            hit_point - hit_sphere.center
        );

        // STEP 5: Calculate direct diffuse lighting.
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

            // STEP 6: Cast a shadow ray.
            bool in_shadow = false;

            for (int j = 0; j < num_shapes; j++) {
                Sphere blocker = spheres[shapes[j].index];

                float shadow_t;

                if (intersect_sphere(
                    hit_point, light_dir, blocker,
                    EPSILON, distance, shadow_t
                )) {
                    in_shadow = true;
                    break;
                }
            }

            // Only add lighting if nothing blocks the light.
            if (!in_shadow) {
                contribution += hit_shape.Kd *
                    light.intensity *
                    (NdotL / distance_sq);
            }
        }

        // STEP 7: Accumulate light from this bounce.
        color += throughput * contribution;

        // STEP 8: Update throughput using reflectivity.
        throughput *= hit_shape.Kr;

        // Stop when reflection has no remaining contribution.
        if (all(throughput == float3(0.0, 0.0, 0.0))) {
            break;
        }

        // STEP 9: Generate the reflected ray.
        ray_origin = hit_point;
        ray_dir = reflect(normalize(ray_dir), normal);
    }

    // STEP 10: Output the final color.
    image[tid] = float4(color, 1.0);
}

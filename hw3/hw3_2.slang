
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

    // The provided solver does not guarantee root order.
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
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  RWTexture2D<float4> image) {

    uint width, height;
    image.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // STEP 1: Generate the camera ray.
    float aspect = float(width) / float(height);

    float x = (2.0 * (float(tid.x) + 0.5)
               / float(width) - 1.0) * aspect;

    float y = 1.0 - 2.0 * (float(tid.y) + 0.5)
              / float(height);

    float3 ray_origin = float3(0.0, 0.0, 0.0);
    float3 ray_dir = float3(x, y, -1.0);

    // STEP 2: Find the closest visible sphere.
    float closest_t = 1e30;
    int closest_shape = -1;

    for (int i = 0; i < num_shapes; i++) {
        Sphere sphere = spheres[shapes[i].index];

        float t;
        if (intersect_sphere(
                ray_origin, ray_dir, sphere,
                EPSILON, closest_t, t)) {

            closest_t = t;
            closest_shape = i;
        }
    }

    // Nothing was hit: use background.
    if (closest_shape == -1) {
        image[tid] = float4(background, 1.0);
        return;
    }

    // STEP 3: Compute intersection point and normal.
    Shape hit_shape = shapes[closest_shape];
    Sphere hit_sphere = spheres[hit_shape.index];

    float3 hit_point = ray_origin + closest_t * ray_dir;

    float3 normal = normalize(
        hit_point - hit_sphere.center
    );

    // STEP 4: Compute diffuse lighting.
    float3 color = float3(0.0, 0.0, 0.0);

    for (int l = 0; l < num_lights; l++) {
        PointLight light = lights[l];

        float3 to_light = light.position - hit_point;
        float distance_sq = dot(to_light, to_light);
        float distance = sqrt(distance_sq);

        if (distance <= EPSILON) {
            continue;
        }

        float3 light_dir = to_light / distance;

        // Lambert's cosine term.
        float NdotL = max(0.0, dot(normal, light_dir));

        if (NdotL <= 0.0) {
            continue;
        }

        // STEP 5: Check whether this light is blocked.
        bool in_shadow = false;

        for (int j = 0; j < num_shapes; j++) {
            Sphere blocker = spheres[shapes[j].index];

            float shadow_t;

            if (intersect_sphere(
                    hit_point, light_dir, blocker,
                    EPSILON, distance, shadow_t)) {

                in_shadow = true;
                break;
            }
        }

        // STEP 6: Add the light only if visible.
        if (!in_shadow) {
            float3 contribution =
                hit_shape.Kd * light.intensity *
                (NdotL / distance_sq);

            color += contribution;
        }
    }

    // STEP 7: Output the final pixel color.
    image[tid] = float4(color, 1.0);
}

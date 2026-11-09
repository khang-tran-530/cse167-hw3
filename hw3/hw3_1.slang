
import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // Step 1: Generate a camera ray through the pixel center.
    float aspect = float(width) / float(height);

    float x = (2.0 * (float(tid.x) + 0.5) / float(width) - 1.0) * aspect;
    float y = 1.0 - 2.0 * (float(tid.y) + 0.5) / float(height);

    float3 ray_origin = float3(0.0, 0.0, 0.0);
    float3 ray_dir = float3(x, y, -1.0);

    // Step 2: Track the closest sphere intersection.
    float closest_t = 1e30;
    float3 pixel_color = background;

    for (int i = 0; i < num_spheres; i++) {
        Sphere sphere = spheres[i];

        float3 oc = ray_origin - sphere.center;

        // Quadratic coefficients.
        float A = dot(ray_dir, ray_dir);
        float B = 2.0 * dot(oc, ray_dir);
        float C = dot(oc, oc) - sphere.radius * sphere.radius;

        float discriminant = B * B - 4.0 * A * C;

        if (discriminant < 0.0) {
            continue;
        }

        float sqrt_d = sqrt(discriminant);

        float t1 = (-B - sqrt_d) / (2.0 * A);
        float t2 = (-B + sqrt_d) / (2.0 * A);

        float t = t1;

        // If the nearest intersection is behind the camera,
        // try the second intersection.
        if (t <= 0.0) {
            t = t2;
        }

        // Keep only the closest positive intersection.
        if (t > 0.0 && t < closest_t) {
            closest_t = t;

            float3 hit_point = ray_origin + t * ray_dir;
            float3 normal = normalize(hit_point - sphere.center);

            // Convert normal components from [-1, 1] to [0, 1].
            pixel_color = (normal + float3(1.0, 1.0, 1.0)) * 0.5;
        }
    }

    // Step 3: Write the final color.
    image[tid] = float4(pixel_color, 1.0);
}

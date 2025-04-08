struct SDFHit
{
    float dist;
    int material_id;
}

struct CameraParams
{
    float3 pos;
    float3 look_at;
    float3 up;
}

struct InputCameraParams
{
    float pos_x, pos_y, pos_z;
    float look_at_x, look_at_y, look_at_z;
    float up_x, up_y, up_z;
}

CameraParams PackCameraParams(InputCameraParams input_params)
{
    CameraParams params;
    params.pos = normalize(float3(input_params.pos_x, input_params.pos_y, input_params.pos_z));
    params.look_at = normalize(float3(input_params.look_at_x, input_params.look_at_y, input_params.look_at_z));
    params.up = normalize(float3(input_params.up_x, input_params.up_y, input_params.up_z));
    return params;
}

SDFHit sdf_sphere(float3 p, float4 center_radius, int material_id)
{
    SDFHit hit;
    hit.dist = length(p - center_radius.xyz) - center_radius.w;
    hit.material_id = material_id;
    return hit;
}

SDFHit sdf_plane(float3 p, float3 point, float3 normal, int material_id)
{
    SDFHit hit;
    hit.dist = dot(p - point, normalize(normal));
    hit.material_id = material_id;
    return hit;
}

SDFHit sdf_union(SDFHit a, SDFHit b)
{
    return (a.dist < b.dist) ? a : b;
}

SDFHit scene_sdf(float3 p, TensorView<float> objects, int obj_count)
{
    SDFHit hit;
    hit.dist = 1e10;
    hit.material_id = -1;

    for (int i = 0; i < obj_count; ++i)
    {
        int base = i * 10;

        int type = int(objects[i, 0]);
        float4 data0 = float4(objects[i, 1], objects[i, 2], objects[i, 3], objects[i, 4]);
        float4 data1 = float4(objects[i, 5], objects[i, 6], objects[i, 7], objects[i, 8]);
        int mat_id = int(objects[i, 9]);

        SDFHit current;

        if (type == 0)
            current = sdf_plane(p, data0.xyz, data1.xyz, mat_id);
        else if (type == 1)
            current = sdf_sphere(p, data0, mat_id);

        hit = sdf_union(hit, current);
    }

    return hit;
}

float raymarch(float3 ro, float3 rd, TensorView<float> objects, int obj_count, out float3 hitPos, out float3 normal, out int material_id)
{
    const int maxSteps = 128;
    const float maxDist = 100.0;
    const float epsilon = 0.001;

    float t = 0.0;

    for (int i = 0; i < maxSteps; ++i)
    {
        float3 p = ro + t * rd;
        SDFHit hit = scene_sdf(p, objects, obj_count);

        if (hit.dist < epsilon)
        {
            hitPos = p;
            material_id = hit.material_id;

            float h = 0.001;
            float3 dx = float3(h, 0, 0);
            float3 dy = float3(0, h, 0);
            float3 dz = float3(0, 0, h);

            normal = normalize(float3(
                scene_sdf(p + dx, objects, obj_count).dist - scene_sdf(p - dx, objects, obj_count).dist,
                scene_sdf(p + dy, objects, obj_count).dist - scene_sdf(p - dy, objects, obj_count).dist,
                scene_sdf(p + dz, objects, obj_count).dist - scene_sdf(p - dz, objects, obj_count).dist
            ));

            return t;
        }

        t += hit.dist;
        if (t > maxDist)
            break;
    }

    hitPos = float3(0.0);
    normal = float3(0.0, 1.0, 0.0);
    material_id = -1;
    return -1.0;
}

[AutoPyBindCUDA]
[CudaKernel]
void render(TensorView<float> output,
            TensorView<float> objects,
            int obj_count,
            InputCameraParams inp_camera_params)
{
    uint3 gid = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (gid.x >= output.size(0) || gid.y >= output.size(1)) return;

    CameraParams cam_params = PackCameraParams(inp_camera_params);

    float2 res = float2(output.size(0), output.size(1));
    float2 uv = (float2(gid.x, gid.y) + 0.5) / res;

    float3 cam_pos = float3(0.0, 0.0, -5.0);
    float3 target = float3(0.0, 0.0, 0.0);
    float3 f = normalize(target - cam_pos);
    float3 up = float3(0.0, 1.0, 0.0);
    float3 r = normalize(cross(f, up));
    float3 u = cross(r, f);

    float fov = 45.0;
    float aspect = res.x / res.y;
    float3 rd = normalize(
        r * (2.0 * uv.x - 1.0) * aspect +
        u * (1.0 - 2.0 * uv.y) +
        f
    );

    float3 pos, normal;
    int mat_id;
    float dist = raymarch(cam_pos, rd, objects, obj_count, pos, normal, mat_id);

    float3 color = (dist > 0.0) ? 0.5 + 0.5 * normal : float3(0.0);
    output[int3(gid.xy, 0)] = color.x;
    output[int3(gid.xy, 1)] = color.y;
    output[int3(gid.xy, 2)] = color.z;
}

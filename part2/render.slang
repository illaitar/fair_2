// Типы материалов
static const int MATERIAL_DIFFUSE = 0;
static const int MATERIAL_MIRROR = 1;
static const int MATERIAL_GLASS = 2;
static const float PI = 3.14159265358979323846;

struct Material
{
    int type;          // Тип материала
    float3 color;      // Цвет материала
    bool has_texture;  // Есть ли текстура
    float param1;      // Дополнительный параметр 1 (например, индекс преломления для стекла)
    float param2;      // Дополнительный параметр 2
};

struct HitInfo
{
    bool didHit;
    float dst;
    float3 hitPoint;
    float3 normal;
    int material_id;
    bool has_texture;
    float texture_offset;
    float texture_width;
    float texture_height;
    float2 uv;
};

struct CameraParams
{
    float3 pos;
    float3 look_at;
    float3 up;
    float vfov;
}

struct InputCameraParams
{
    float pos_x, pos_y, pos_z;
    float look_at_x, look_at_y, look_at_z;
    float up_x, up_y, up_z;
    float vfov;
}

CameraParams PackCameraParams(InputCameraParams input_params)
{
    CameraParams params;
    params.pos = float3(input_params.pos_x, input_params.pos_y, input_params.pos_z);
    params.look_at = float3(input_params.look_at_x, input_params.look_at_y, input_params.look_at_z);
    params.up = normalize(float3(input_params.up_x, input_params.up_y, input_params.up_z));
    params.vfov = input_params.vfov;
    return params;
}

float2 sphere_uv(float3 p, float3 center)
{
    float3 dir = normalize(p - center);
    float2 uv;
    uv.x = 0.5 + atan2(dir.z, dir.x) / (2.0 * 3.1415926);
    uv.y = 0.5 - asin(dir.y) / 3.1415926;
    return uv;
}

float2 plane_uv(float3 p, float3 point, float3 normal, float tiling_scale = 0.01, float2 offset = float2(0.1, 0.1))
{
    float3 tangent, bitangent;
    float3 up = float3(0, 1, 0);
    
    if (abs(dot(normal, up)) > 0.99)
        tangent = normalize(cross(normal, float3(1, 0, 0)));
    else
        tangent = normalize(cross(normal, up));
    
    bitangent = normalize(cross(normal, tangent));
    
    float3 local_p = p - point;
    float2 uv;
    uv.x = dot(local_p, tangent) * tiling_scale;
    uv.y = dot(local_p, bitangent) * tiling_scale;
    
    // Добавляем смещение перед применением frac
    return frac(uv + offset);
}


HitInfo sdf_sphere(float3 p, float4 center_radius, int material_id, bool has_texture, float texture_offset, float texture_width, float texture_height)
{
    HitInfo hit;
    hit.dst = length(p - center_radius.xyz) - center_radius.w;
    hit.didHit = false; // Will be set later during raymarching
    hit.hitPoint = float3(0.0);
    hit.normal = float3(0.0);
    hit.material_id = material_id;
    hit.has_texture = has_texture;
    hit.texture_offset = texture_offset;
    hit.texture_width = texture_width;
    hit.texture_height = texture_height;
    hit.uv = sphere_uv(p, center_radius.xyz);
    return hit;
}

HitInfo sdf_plane(float3 p, float3 point, float3 normal, int material_id, bool has_texture, float texture_offset, float texture_width, float texture_height, float tiling_scale = 0.01, float2 offset = float2(0.1, 0.1))
{
    HitInfo hit;
    hit.dst = dot(p - point, normalize(normal));
    hit.didHit = false; // Will be set later during raymarching
    hit.hitPoint = float3(0.0);
    hit.normal = normalize(normal);
    hit.material_id = material_id;
    hit.has_texture = has_texture;
    hit.texture_offset = texture_offset;
    hit.texture_width = texture_width;
    hit.texture_height = texture_height;
    hit.uv = plane_uv(p, point, normal, tiling_scale, offset);
    return hit;
}

HitInfo sdf_union(HitInfo a, HitInfo b)
{
    return (a.dst < b.dst) ? a : b;
}

HitInfo scene_sdf(float3 p, TensorView<float> objects, int obj_count, float plane_tiling_scale = 0.01, float2 plane_offset = float2(0.1, 0.1))
{
    HitInfo hit;
    hit.dst = 1e10;
    hit.didHit = false;
    hit.hitPoint = float3(0.0);
    hit.normal = float3(0.0, 1.0, 0.0);
    hit.material_id = -1;
    hit.has_texture = false;
    hit.texture_offset = 0;
    hit.texture_width = 0;
    hit.texture_height = 0;
    hit.uv = float2(0, 0);

    for (int i = 0; i < obj_count; ++i)
    {
        int type = int(objects[i, 0]);
        float4 data0 = float4(objects[i, 1], objects[i, 2], objects[i, 3], objects[i, 4]);
        float4 data1 = float4(objects[i, 5], objects[i, 6], objects[i, 7], objects[i, 8]);
        int mat_id = int(objects[i, 9]);
        
        // Параметры текстуры
        bool has_texture = objects[i, 10] > 0.5;
        float texture_offset = objects[i, 11];
        float texture_width = objects[i, 12];
        float texture_height = objects[i, 13];

        HitInfo current;

        if (type == 0)
            current = sdf_plane(p, data0.xyz, data1.xyz, mat_id, has_texture, texture_offset, texture_width, texture_height, plane_tiling_scale, plane_offset);
        else if (type == 1)
            current = sdf_sphere(p, data0, mat_id, has_texture, texture_offset, texture_width, texture_height);

        hit = sdf_union(hit, current);
    }

    return hit;
}

// Функция для извлечения текстуры из буфера
float3 sample_texture(TensorView<float> textures, float texture_offset, float texture_width, float texture_height, float2 uv)
{
    uv = float2(frac(uv.x), frac(uv.y));
    int x = int(uv.x * texture_width);
    int y = int(uv.y * texture_height);
    x = min(max(x, 0), int(texture_width) - 1);
    y = min(max(y, 0), int(texture_height) - 1);
    int pixel_index = int(texture_offset) + y * int(texture_width) + x;
    
    if (pixel_index >= 0 && pixel_index < textures.size(0))
        return float3(textures[pixel_index, 0], textures[pixel_index, 1], textures[pixel_index, 2]);
    else
        return float3(1, 0, 1); // Пурпурный для отладки
}

struct Ray
{
    float3 origin;
    float3 dir;
}

HitInfo raymarch(Ray ray, TensorView<float> objects, int obj_count, float plane_tiling_scale = 0.01, float2 plane_offset = float2(0.1, 0.1))
{
    const int maxSteps = 256;
    const float maxDist = 100.0;
    const float epsilon = 0.0001;

    float t = 0.0;
    
    HitInfo finalHit;
    finalHit.didHit = false;
    finalHit.dst = -1.0;
    finalHit.hitPoint = float3(0.0);
    finalHit.normal = float3(0.0, 1.0, 0.0);
    finalHit.material_id = -1;
    finalHit.has_texture = false;
    finalHit.texture_offset = 0;
    finalHit.texture_width = 0;
    finalHit.texture_height = 0;
    finalHit.uv = float2(0, 0);

    for (int i = 0; i < maxSteps; ++i)
    {
        float3 p = ray.origin + t * ray.dir;
        HitInfo hit = scene_sdf(p, objects, obj_count, plane_tiling_scale, plane_offset);

        if (hit.dst < epsilon)
        {
            finalHit = hit;
            finalHit.didHit = true;
            finalHit.dst = t;
            finalHit.hitPoint = p;

            // Calculate normal using finite difference
            float h = 0.001;
            float3 dx = float3(h, 0, 0);
            float3 dy = float3(0, h, 0);
            float3 dz = float3(0, 0, h);

            finalHit.normal = normalize(float3(
                scene_sdf(p + dx, objects, obj_count, plane_tiling_scale, plane_offset).dst - scene_sdf(p - dx, objects, obj_count, plane_tiling_scale, plane_offset).dst,
                scene_sdf(p + dy, objects, obj_count, plane_tiling_scale, plane_offset).dst - scene_sdf(p - dy, objects, obj_count, plane_tiling_scale, plane_offset).dst,
                scene_sdf(p + dz, objects, obj_count, plane_tiling_scale, plane_offset).dst - scene_sdf(p - dz, objects, obj_count, plane_tiling_scale, plane_offset).dst
            ));

            return finalHit;
        }

        t += hit.dst;
        if (t > maxDist)
            break;
    }

    return finalHit; // No hit found
}

// Функция для получения материала по его id
Material get_material(TensorView<float> materials, int material_id, int materials_count)
{
    Material mat;
    
    // По умолчанию серый диффузный материал
    mat.type = MATERIAL_DIFFUSE;
    mat.color = float3(0.8, 0.8, 0.8);
    mat.has_texture = false;
    mat.param1 = 0.0;
    mat.param2 = 0.0;
    
    // Проверяем, что материал существует
    if (material_id >= 0 && material_id < materials_count) {
        mat.type = int(materials[material_id, 0]);
        mat.color = float3(
            materials[material_id, 1], 
            materials[material_id, 2], 
            materials[material_id, 3]
        );
        mat.has_texture = materials[material_id, 4] > 0.0;
        mat.param1 = materials[material_id, 5];
        mat.param2 = materials[material_id, 6];
    }
    
    return mat;
}

// Функция для отражения луча
float3 reflect_ray(float3 incident, float3 normal)
{
    return incident - 2.0 * dot(incident, normal) * normal;
}

// Функция для преломления луча
float3 refract_ray(float3 incident, float3 normal, float eta)
{
    float cos_i = -dot(normal, incident);
    float cos_t2 = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    
    if (cos_t2 > 0.0) {
        return eta * incident + (eta * cos_i - sqrt(cos_t2)) * normal;
    }
    
    return reflect_ray(incident, normal);
}

// Функция для вращения вектора вокруг осей
float3 rotate_direction(float3 dir, float rot_x, float rot_y, float rot_z)
{
    // Вращение вокруг оси X (наклон вверх/вниз)
    float cos_x = cos(rot_x);
    float sin_x = sin(rot_x);
    float3 dir_x = float3(
        dir.x,
        dir.y * cos_x - dir.z * sin_x,
        dir.y * sin_x + dir.z * cos_x
    );
    
    // Вращение вокруг оси Y (поворот влево/вправо)
    float cos_y = cos(rot_y);
    float sin_y = sin(rot_y);
    float3 dir_xy = float3(
        dir_x.x * cos_y + dir_x.z * sin_y,
        dir_x.y,
        -dir_x.x * sin_y + dir_x.z * cos_y
    );
    
    // Вращение вокруг оси Z (наклон влево/вправо)
    float cos_z = cos(rot_z);
    float sin_z = sin(rot_z);
    float3 dir_xyz = float3(
        dir_xy.x * cos_z - dir_xy.y * sin_z,
        dir_xy.x * sin_z + dir_xy.y * cos_z,
        dir_xy.z
    );
    
    return normalize(dir_xyz);
}

// Функция для преобразования направления луча в UV-координаты на сферической панораме
float2 direction_to_equirect_uv(float3 direction, float rot_x = 0.0, float rot_y = 0.0, float rot_z = 0.0)
{
    // Нормализация направления
    float3 dir = normalize(direction);
    
    // Применяем вращение, если нужно
    if (rot_x != 0.0 || rot_y != 0.0 || rot_z != 0.0) {
        dir = rotate_direction(dir, rot_x, rot_y, rot_z);
    }
    
    // Преобразование направления в сферические координаты
    float phi = atan2(dir.z, dir.x);
    float theta = asin(dir.y);
    
    // Преобразование в UV-координаты [0,1]
    float u = (phi + PI) / (2.0 * PI);
    float v = (theta + PI / 2.0) / PI;
    
    return float2(u, v);
}

// Функция для семплирования сферической панорамы
float3 sample_environment(TensorView<float> env_texture, float3 direction, int env_width, int env_height, 
                          float rot_x = 0.0, float rot_y = 0.0, float rot_z = 0.0)
{
    // Получаем UV-координаты из направления луча с учетом вращения
    float2 uv = direction_to_equirect_uv(direction, rot_x, rot_y, rot_z);
    
    // Преобразуем UV в координаты текстуры
    int x = int(uv.x * env_width) % env_width;
    int y = int(uv.y * env_height) % env_height;
    
    // Убедимся, что координаты положительные
    if (x < 0) x += env_width;
    if (y < 0) y += env_height;
    
    // Получаем пиксель из текстуры
    int pixel_index = y * env_width + x;
    if (pixel_index >= 0 && pixel_index < env_texture.size(0) && env_width > 0 && env_height > 0) {
        return float3(
            env_texture[pixel_index, 0],
            env_texture[pixel_index, 1],
            env_texture[pixel_index, 2]
        );
    } else {
        // Если текстура недоступна, возвращаем градиентное небо
        float t = 0.5 * (direction.y + 1.0);
        return float3(1.0, 1.0, 1.0) * (1.0 - t) + float3(0.5, 0.7, 1.0) * t;
    }
}

[AutoPyBindCUDA]
[CudaKernel]
void render(TensorView<float> output,
            TensorView<float> objects,
            int obj_count,
            InputCameraParams inp_camera_params,
            TensorView<float> textures,
            TensorView<float> materials,
            int materials_count,
            TensorView<float> env_texture,
            int env_width,
            int env_height,
            bool path_tracing,
            uint input_seed,
            uint path_tracing_iters = 1,
            uint max_bounces = 10,
            float env_rot_x = 0.0,
            float env_rot_y = 0.0,
            float env_rot_z = 0.0,
            float plane_tiling_scale = 0.01,
            float plane_offset_x = 0.1,
            float plane_offset_y = 0.1)
{
    uint3 gid = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (gid.x >= output.size(0) || gid.y >= output.size(1)) return;

    float2 plane_offset = float2(plane_offset_x, plane_offset_y);
    
    CameraParams cam_params = PackCameraParams(inp_camera_params);

    // Screen dimensions
    float2 res = float2(output.size(0), output.size(1));
    float aspect = res.x / res.y;
    
    // Convert vertical FOV from degrees to radians
    float vfov_rad = radians(cam_params.vfov);
    
    // Camera setup with orthonormal basis
    float3 cam_pos = cam_params.pos;
    float3 cam_forward = normalize(cam_params.look_at - cam_pos);
    float3 cam_right = normalize(cross(cam_forward, cam_params.up));
    float3 cam_up = normalize(cross(cam_right, cam_forward));
    
    // Screen coordinates in [-1,1] range
    float x = (2.0 * (float(gid.x) + 0.5) / res.x - 1.0);
    float y = (1.0 - 2.0 * (float(gid.y) + 0.5) / res.y);
    
    // Calculate camera image plane distance for FOV
    float image_plane_dist = 1.0 / tan(vfov_rad * 0.5);
    
    // Calculate ray direction 
    float3 rd = normalize(
        cam_forward * image_plane_dist + 
        cam_right * x * aspect + 
        cam_up * y
    );

    // Итеративная трассировка лучей с ограниченной глубиной
    int MAX_BOUNCES = max_bounces; // Максимальное количество отражений/преломлений

    float3 mean_result_color = float3(0.0, 0.0, 0.0);

    for (int i = 0; i < path_tracing_iters; i++){
        // Начальные значения луча
        Ray ray;
        ray.origin = cam_pos;
        ray.dir = rd;
        float3 result_color = float3(0.0, 0.0, 0.0);
        float3 attenuation = float3(1.0, 1.0, 1.0); // Текущее ослабление луча
        
        // Инициализируем seed для случайных чисел
        uint seed = uint(gid.x + gid.y * output.size(0)) * 719393 + 1337 * i;
        
        // Итеративная трассировка с ограниченной глубиной
        for (int bounce = 0; bounce < MAX_BOUNCES; bounce++)
        {
            HitInfo hit = raymarch(ray, objects, obj_count, plane_tiling_scale, plane_offset);
            
            // Если луч не попал в объект, добавляем цвет фона и выходим
            if (!hit.didHit)
            {
                float3 sky_color = sample_environment(env_texture, ray.dir, env_width, env_height, 
                                                     env_rot_x, env_rot_y, env_rot_z);
                result_color = result_color + attenuation * sky_color;
                break;
            }
            
            Material material = get_material(materials, hit.material_id, materials_count);
            
            // Базовый цвет материала (с учетом текстуры, если она есть)
            float3 material_color = material.color;
            if (material.has_texture && hit.has_texture) {
                material_color = sample_texture(textures, hit.texture_offset, hit.texture_width, hit.texture_height, hit.uv);
            }
            
            // Обработка в зависимости от типа материала
            if (material.type == MATERIAL_DIFFUSE)
            {
                if (path_tracing)
                {
                    float3 random_dir = random_hemisphere_direction(hit.normal, seed);
                    
                    ray.origin = hit.hitPoint + hit.normal * 0.001; // Небольшое смещение
                    ray.dir = random_dir;
                    
                    // Ослабляем цвет по закону Ламберта (косинус угла между нормалью и направлением)
                    float cos_theta = dot(hit.normal, random_dir);
                    attenuation = attenuation * material_color * cos_theta * 2;
                    float p = max(attenuation.r, max(attenuation.g, attenuation.b));
                }
                else
                {
                    float diffuse = 1.0;
                    float3 ambient = float3(0.1, 0.1, 0.1);
                    
                    result_color = result_color + attenuation * material_color * (ambient + diffuse);
                    break;
                }
            }
            else if (material.type == MATERIAL_MIRROR)
            {
                // Отражение для зеркала
                float3 reflect_dir = reflect_ray(ray.dir, hit.normal);
                
                ray.origin = hit.hitPoint + hit.normal * 0.001; // Небольшое смещение для избежания самопересечения
                ray.dir = reflect_dir;
                
                attenuation = attenuation * material_color;
                
                // В режиме обычного рейтрейсинга добавляем блик
                if (!path_tracing) {
                    result_color = result_color + attenuation * float3(1.0, 1.0, 0.9) * 0.2;
                }
            }
            else if (material.type == MATERIAL_GLASS)
            {
                // Стеклянный материал (преломление с учетом френеля)
                float refraction_index = max(1.01, material.param1);
                if (refraction_index < 1.01) refraction_index = 1.5; // По умолчанию для стекла
                
                // Вычисляем коэффициент Френеля
                float fresnel = 0.1 + 0.9 * pow(1.0 - max(0.0, -dot(ray.dir, hit.normal)), 5.0);
                
                // Решаем, будет ли отражение или преломление на основе коэффициента Френеля
                if (bounce == MAX_BOUNCES - 1 || RandomValue(seed) < fresnel)
                {
                    // Отражение (как у зеркала)
                    float3 reflect_dir = reflect_ray(ray.dir, hit.normal);
                    ray.origin = hit.hitPoint + hit.normal * 0.001;
                    ray.dir = reflect_dir;
                }
                else
                {
                    // Преломление
                    float3 refract_dir = refract_ray(ray.dir, hit.normal, 1.0 / refraction_index);
                    ray.origin = hit.hitPoint - hit.normal * 0.001;
                    ray.dir = refract_dir;
                }
                
                // Учитываем цвет стекла в ослаблении
                attenuation = attenuation * material_color;
            }
        }
        
        result_color = min(result_color, float3(1.0, 1.0, 1.0));
        mean_result_color += result_color;
    }
    mean_result_color = mean_result_color / path_tracing_iters;
    output[int3(gid.xy, 0)] = mean_result_color.x;
    output[int3(gid.xy, 1)] = mean_result_color.y;
    output[int3(gid.xy, 2)] = mean_result_color.z;
}

// Функция для получения дробной части числа
float frac(float x)
{
    return x - floor(x);
}

// Простая функция псевдо-случайных чисел (для вероятностных решений)
float rand(float seed)
{
    float a = sin(seed) * 43758.5453;
    return frac(a);
}

uint NextRandom(inout uint state)
{
    state = state * 747796405 + 2891336453;
    uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    result = (result >> 22) ^ result;
    return result;
}

float RandomValue(inout uint state)
{
    return NextRandom(state) / 4294967295.0; // 2^32 - 1
}

// Random value in normal distribution (with mean=0 and sd=1)
float RandomValueNormalDistribution(inout uint state)
{
    // Thanks to https://stackoverflow.com/a/6178290
    float theta = 2 * 3.1415926 * RandomValue(state);
    float rho = sqrt(-2 * log(RandomValue(state)));
    return rho * cos(theta);
}

// Calculate a random direction
float3 RandomDirection(inout uint state)
{
    // Thanks to https://math.stackexchange.com/a/1585996
    float x = RandomValueNormalDistribution(state);
    float y = RandomValueNormalDistribution(state);
    float z = RandomValueNormalDistribution(state);
    return normalize(float3(x, y, z));
}

float2 RandomPointInCircle(inout uint rngState)
{
    float angle = RandomValue(rngState) * 2 * PI;
    float2 pointOnCircle = float2(cos(angle), sin(angle));
    return pointOnCircle * sqrt(RandomValue(rngState));
}

float2 mod2(float2 x, float2 y)
{
    return x - y * floor(x/y);
}

float3 random_hemisphere_direction(float3 normal, uint seed){
    float3 dir = RandomDirection(seed);
    return dir * sign(dot(normal, dir));
}
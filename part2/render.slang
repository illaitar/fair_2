// Типы материалов
static const int MATERIAL_DIFFUSE = 0;
static const int MATERIAL_MIRROR = 1;
static const int MATERIAL_GLASS = 2;

struct Material
{
    int type;          // Тип материала
    float3 color;      // Цвет материала
    bool has_texture;  // Есть ли текстура
    float param1;      // Дополнительный параметр 1 (например, индекс преломления для стекла)
    float param2;      // Дополнительный параметр 2
};

struct SDFHit
{
    float dist;
    int material_id;
    bool has_texture;
    float texture_offset;
    float texture_width;
    float texture_height;
    float2 uv; // Текстурные координаты
}

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

// Функция для генерации UV-координат для сферы
float2 sphere_uv(float3 p, float3 center)
{
    float3 dir = normalize(p - center);
    float2 uv;
    uv.x = 0.5 + atan2(dir.z, dir.x) / (2.0 * 3.1415926);
    uv.y = 0.5 - asin(dir.y) / 3.1415926;
    return uv;
}

// Функция для генерации UV-координат для плоскости
float2 plane_uv(float3 p, float3 point, float3 normal)
{
    // Создаем локальную систему координат на плоскости
    float3 tangent, bitangent;
    float3 up = float3(0, 1, 0);
    
    // Если нормаль параллельна вектору up, используем другой вектор
    if (abs(dot(normal, up)) > 0.99)
        tangent = normalize(cross(normal, float3(1, 0, 0)));
    else
        tangent = normalize(cross(normal, up));
    
    bitangent = normalize(cross(normal, tangent));
    
    // Проецируем точку на плоскость
    float3 local_p = p - point;
    float2 uv;
    uv.x = dot(local_p, tangent) * 0.2; // Масштабируем для повторения текстуры
    uv.y = dot(local_p, bitangent) * 0.2;
    
    // Возвращаем дробную часть для повторения текстуры
    return float2(frac(uv.x), frac(uv.y));
}

SDFHit sdf_sphere(float3 p, float4 center_radius, int material_id, bool has_texture, float texture_offset, float texture_width, float texture_height)
{
    SDFHit hit;
    hit.dist = length(p - center_radius.xyz) - center_radius.w;
    hit.material_id = material_id;
    hit.has_texture = has_texture;
    hit.texture_offset = texture_offset;
    hit.texture_width = texture_width;
    hit.texture_height = texture_height;
    hit.uv = sphere_uv(p, center_radius.xyz);
    return hit;
}

SDFHit sdf_plane(float3 p, float3 point, float3 normal, int material_id, bool has_texture, float texture_offset, float texture_width, float texture_height)
{
    SDFHit hit;
    hit.dist = dot(p - point, normalize(normal));
    hit.material_id = material_id;
    hit.has_texture = has_texture;
    hit.texture_offset = texture_offset;
    hit.texture_width = texture_width;
    hit.texture_height = texture_height;
    hit.uv = plane_uv(p, point, normal);
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

        SDFHit current;

        if (type == 0)
            current = sdf_plane(p, data0.xyz, data1.xyz, mat_id, has_texture, texture_offset, texture_width, texture_height);
        else if (type == 1)
            current = sdf_sphere(p, data0, mat_id, has_texture, texture_offset, texture_width, texture_height);

        hit = sdf_union(hit, current);
    }

    return hit;
}

// Функция для извлечения текстуры из буфера
float3 sample_texture(TensorView<float> textures, float texture_offset, float texture_width, float texture_height, float2 uv)
{
    // Убедимся, что UV находятся в диапазоне [0,1]
    uv = float2(frac(uv.x), frac(uv.y));
    
    // Вычисляем координаты пикселя
    int x = int(uv.x * texture_width);
    int y = int(uv.y * texture_height);
    
    // Клампим к размерам текстуры
    x = min(max(x, 0), int(texture_width) - 1);
    y = min(max(y, 0), int(texture_height) - 1);
    
    // Вычисляем индекс пикселя в буфере
    int pixel_index = int(texture_offset) + y * int(texture_width) + x;
    
    // Безопасная проверка границ
    if (pixel_index >= 0 && pixel_index < textures.size(0))
        return float3(textures[pixel_index, 0], textures[pixel_index, 1], textures[pixel_index, 2]);
    else
        return float3(1, 0, 1); // Пурпурный для отладки
}

float raymarch(float3 ro, float3 rd, TensorView<float> objects, int obj_count, out float3 hitPos, out float3 normal, out int material_id, out float2 uv, out bool has_texture, out float texture_offset, out float texture_width, out float texture_height)
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
            uv = hit.uv;
            has_texture = hit.has_texture;
            texture_offset = hit.texture_offset;
            texture_width = hit.texture_width;
            texture_height = hit.texture_height;

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
    uv = float2(0, 0);
    has_texture = false;
    texture_offset = 0;
    texture_width = 0;
    texture_height = 0;
    return -1.0;
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
    
    // Полное внутреннее отражение
    return reflect_ray(incident, normal);
}

// Функция для расчета цвета при попадании в объект
float3 shade_hit(float3 pos, float3 normal, float3 ray_dir, Material material, 
                float2 uv, TensorView<float> textures, float texture_offset, 
                float texture_width, float texture_height, bool has_texture,
                TensorView<float> objects, int obj_count, float3 cam_pos)
{
    // Базовый цвет материала
    float3 material_color = material.color;
    
    // Если есть текстура, используем её
    if (material.has_texture && has_texture) {
        material_color = sample_texture(textures, texture_offset, texture_width, texture_height, uv);
    }
    
    // Обработка в зависимости от типа материала
    if (material.type == MATERIAL_DIFFUSE) {
        // Диффузный материал (модель Ламберта)
        float3 light_dir = normalize(float3(1.0, 1.0, 1.0));
        float diffuse = max(0.2, dot(normal, light_dir));
        float3 ambient = float3(0.1, 0.1, 0.1);
        
        return material_color * (ambient + diffuse);
    }
    else if (material.type == MATERIAL_MIRROR) {
        // Зеркальный материал (отражение)
        float3 reflect_dir = reflect_ray(ray_dir, normal);
        
        // Выполняем упрощенное отражение (без рекурсии)
        // В реальном рейтрейсере здесь был бы рекурсивный вызов
        float3 reflected_color = float3(0.3, 0.3, 0.3);  // Приближение для отраженного окружения
        
        // Добавляем имитацию отраженного света в направлении неба
        if (reflect_dir.y > 0) {
            // Линейная интерполяция: a * (1 - t) + b * t
            reflected_color = reflected_color * (1.0 - reflect_dir.y) + float3(0.7, 0.8, 1.0) * reflect_dir.y;
        }
        
        return material_color * reflected_color;
    }
    else if (material.type == MATERIAL_GLASS) {
        // Стеклянный материал (преломление)
        float refraction_index = max(1.01, material.param1);  // Индекс преломления
        if (refraction_index < 1.01) refraction_index = 1.5;  // По умолчанию для стекла
        
        float3 refract_dir = refract_ray(ray_dir, normal, 1.0 / refraction_index);
        
        // Упрощенная модель преломления без рекурсии
        float3 refracted_color = float3(0.6, 0.6, 0.6);
        
        // Имитация преломления света
        float fresnel = 0.1 + 0.9 * pow(1.0 - max(0.0, -dot(ray_dir, normal)), 5.0);
        
        // Блики на поверхности
        float3 reflect_dir = reflect_ray(ray_dir, normal);
        float spec = pow(max(dot(reflect_dir, normalize(cam_pos - pos)), 0.0), 50.0);
        
        // Линейная интерполяция: a * (1 - t) + b * t
        return material_color * refracted_color * (1.0 - spec * fresnel) + float3(1.0) * (spec * fresnel);
    }
    
    // Для неизвестного типа материала возвращаем базовый цвет
    return material_color;
}

[AutoPyBindCUDA]
[CudaKernel]
void render(TensorView<float> output,
            TensorView<float> objects,
            int obj_count,
            InputCameraParams inp_camera_params,
            TensorView<float> textures,
            TensorView<float> materials,
            int materials_count)
{
    uint3 gid = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (gid.x >= output.size(0) || gid.y >= output.size(1)) return;

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

    float3 pos, normal;
    int mat_id;
    float2 uv;
    bool has_texture;
    float texture_offset, texture_width, texture_height;
    
    float dist = raymarch(cam_pos, rd, objects, obj_count, pos, normal, mat_id, uv, has_texture, texture_offset, texture_width, texture_height);

    float3 color;
    
    if (dist > 0.0) {
        // Получаем материал объекта
        Material material = get_material(materials, mat_id, materials_count);
        
        // Вычисляем цвет с учетом материала и освещения
        color = shade_hit(
            pos, normal, rd, material,
            uv, textures, texture_offset, texture_width, texture_height, has_texture,
            objects, obj_count, cam_pos
        );
    } else {
        // Фон (небо)
        float t = 0.5 * (rd.y + 1.0);
        // Линейная интерполяция: a * (1 - t) + b * t
        color = float3(1.0, 1.0, 1.0) * (1.0 - t) + float3(0.5, 0.7, 1.0) * t;
    }
    
    // Простая коррекция гаммы
    color = pow(color, float3(1.0/2.2));
    
    output[int3(gid.xy, 0)] = color.x;
    output[int3(gid.xy, 1)] = color.y;
    output[int3(gid.xy, 2)] = color.z;
}

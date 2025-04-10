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

float2 plane_uv(float3 p, float3 point, float3 normal)
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
    uv.x = dot(local_p, tangent) * 0.01;
    uv.y = dot(local_p, bitangent) * 0.01;
    return float2(frac(uv.x), frac(uv.y));
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

HitInfo sdf_plane(float3 p, float3 point, float3 normal, int material_id, bool has_texture, float texture_offset, float texture_width, float texture_height)
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
    hit.uv = plane_uv(p, point, normal);
    return hit;
}

HitInfo sdf_union(HitInfo a, HitInfo b)
{
    return (a.dst < b.dst) ? a : b;
}

HitInfo scene_sdf(float3 p, TensorView<float> objects, int obj_count)
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

HitInfo raymarch(Ray ray, TensorView<float> objects, int obj_count)
{
    const int maxSteps = 128;
    const float maxDist = 100.0;
    const float epsilon = 0.001;

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
        HitInfo hit = scene_sdf(p, objects, obj_count);

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
                scene_sdf(p + dx, objects, obj_count).dst - scene_sdf(p - dx, objects, obj_count).dst,
                scene_sdf(p + dy, objects, obj_count).dst - scene_sdf(p - dy, objects, obj_count).dst,
                scene_sdf(p + dz, objects, obj_count).dst - scene_sdf(p - dz, objects, obj_count).dst
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

// Функция для преобразования направления луча в UV-координаты на сферической панораме
float2 direction_to_equirect_uv(float3 direction)
{
    // Нормализация направления
    float3 dir = normalize(direction);
    
    // Преобразование направления в сферические координаты
    float phi = atan2(dir.z, dir.x);
    float theta = asin(dir.y);
    
    // Преобразование в UV-координаты [0,1]
    float u = (phi + 3.1415926) / (2.0 * 3.1415926);
    float v = (theta + 3.1415926 / 2.0) / 3.1415926;
    
    return float2(u, v);
}

// Функция для семплирования сферической панорамы
float3 sample_environment(TensorView<float> env_texture, float3 direction, int env_width, int env_height)
{
    // Получаем UV-координаты из направления луча
    float2 uv = direction_to_equirect_uv(direction);
    
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

// Функция для генерации случайного направления в полусфере вокруг нормали
float3 random_hemisphere_direction(float3 normal, uint seed)
{
    // Генерируем два случайных числа для сферических координат
    float rand1 = RandomValue(seed);
    float rand2 = RandomValue(seed);
    
    // Преобразуем в направление в локальной системе координат полусферы
    float cos_theta = sqrt(1.0 - rand1);
    float sin_theta = sqrt(rand1);
    float phi = 2.0 * 3.1415926 * rand2;
    
    float3 direction = float3(
        sin_theta * cos(phi),
        cos_theta,               // Y is up in local space
        sin_theta * sin(phi)
    );
    
    // Создаем ортонормированный базис вокруг нормали
    float3 up = abs(normal.y) > 0.999 ? float3(1, 0, 0) : float3(0, 1, 0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    
    // Преобразуем из локальной системы координат в мировую
    return normalize(
        tangent * direction.x +
        normal * direction.y +
        bitangent * direction.z
    );
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
            uint seed)
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

    // Итеративная трассировка лучей с ограниченной глубиной
    const int MAX_BOUNCES = 5; // Максимальное количество отражений/преломлений
    
    // Начальные значения луча
    Ray ray;
    ray.origin = cam_pos;
    ray.dir = rd;
    float3 result_color = float3(0.0, 0.0, 0.0);
    float3 attenuation = float3(1.0, 1.0, 1.0); // Текущее ослабление луча
    
    // Инициализируем seed для случайных чисел
    // uint seed = uint(gid.x + gid.y * output.size(0)) * 719393 + 1337;
    
    // Итеративная трассировка с ограниченной глубиной
    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++)
    {
        // Трассируем текущий луч
        HitInfo hit = raymarch(ray, objects, obj_count);
        
        // Если луч не попал в объект, добавляем цвет фона и выходим
        if (!hit.didHit)
        {
            // Используем сферическую панораму как фон
            float3 sky_color = sample_environment(env_texture, ray.dir, env_width, env_height);
            
            // Добавляем цвет фона с учетом текущего ослабления
            result_color = result_color + attenuation * sky_color;
            break;
        }
        
        // Получаем материал объекта
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
                // Path tracing - случайное направление в полусфере вокруг нормали
                float3 random_dir = random_hemisphere_direction(hit.normal, seed);
                
                // Настраиваем следующий луч
                ray.origin = hit.hitPoint + hit.normal * 0.001; // Небольшое смещение
                ray.dir = random_dir;
                
                // Ослабляем цвет по закону Ламберта (косинус угла между нормалью и направлением)
                float cos_theta = max(0.0, dot(hit.normal, random_dir));
                attenuation = attenuation * material_color * cos_theta;
                
                // Для оптимизации добавляем ambient для первого отражения
                if (bounce == 0) {
                    float3 ambient = float3(0.1, 0.1, 0.1);
                    result_color = result_color + attenuation * ambient;
                }
            }
            else
            {
                // Обычный рейтрейсинг с простым диффузным затенением
                float3 light_dir = normalize(float3(1.0, 1.0, 1.0));
                float diffuse = max(0.2, dot(hit.normal, light_dir));
                float3 ambient = float3(0.1, 0.1, 0.1);
                
                // Добавляем диффузный цвет с учетом текущего ослабления
                result_color = result_color + attenuation * material_color * (ambient + diffuse);
                break; // Конец трассировки для диффузных поверхностей в обычном режиме
            }
        }
        else if (material.type == MATERIAL_MIRROR)
        {
            // Отражение для зеркала
            float3 reflect_dir = reflect_ray(ray.dir, hit.normal);
            
            // Настраиваем следующий луч как отраженный
            ray.origin = hit.hitPoint + hit.normal * 0.001; // Небольшое смещение для избежания самопересечения
            ray.dir = reflect_dir;
            
            // Учитываем цвет зеркала в ослаблении
            attenuation = attenuation * material_color;
            
            // В режиме обычного рейтрейсинга добавляем блик
            if (!path_tracing) {
                float3 light_dir = normalize(float3(1.0, 1.0, 1.0));
                float light_reflection = pow(max(0.0, dot(reflect_dir, light_dir)), 8.0);
                result_color = result_color + attenuation * float3(1.0, 1.0, 0.9) * light_reflection * 0.2;
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
        else
        {
            // Неизвестный материал (обрабатываем как диффузный)
            if (path_tracing)
            {
                // Path tracing - случайное направление в полусфере
                float3 random_dir = random_hemisphere_direction(hit.normal, seed);
                
                // Настраиваем следующий луч
                ray.origin = hit.hitPoint + hit.normal * 0.001;
                ray.dir = random_dir;
                
                // Ослабляем цвет по закону Ламберта
                float cos_theta = max(0.0, dot(hit.normal, random_dir));
                attenuation = attenuation * material_color * cos_theta;
            }
            else
            {
                // Обычный режим - просто берем цвет материала
                result_color = result_color + attenuation * material_color;
                break;
            }
        }
        
        // Проверка на слишком сильное ослабление (оптимизация)
        if (max(max(attenuation.x, attenuation.y), attenuation.z) < 0.01)
            break;
    }
    
    // Простая коррекция гаммы
    result_color = pow(result_color, float3(1.0/2.2));
    
    // Клампим цвет для безопасности
    result_color = min(result_color, float3(1.0, 1.0, 1.0));
    
    output[int3(gid.xy, 0)] = result_color.x;
    output[int3(gid.xy, 1)] = result_color.y;
    output[int3(gid.xy, 2)] = result_color.z;
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

float RandomValue(inout uint state)
{
    state = state * 747796405 + 2891336453;
    uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    result = (result >> 22) ^ result;
    return result / 4294967295.0;
}
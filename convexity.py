import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Tuple, Dict

def curve_convexity_metric_6_points(points: np.ndarray) -> Tuple[float, Dict]:
    """
    Вычисляет метрику выпуклости/вогнутости для кривой, заданной 6 точками.
    
    Параметры:
    -----------
    points : np.ndarray
        Массив формы (6, 2) с координатами точек в порядке следования вдоль кривой
    
    Возвращает:
    -----------
    Tuple[float, Dict]: 
        - float: метрика выпуклости (положительная = выпуклая, отрицательная = вогнутая)
        - Dict: дополнительная информация о вычислениях
    """
    
    def normalize_points(points):
        """Нормализация точек для устранения влияния масштаба"""
        centroid = np.mean(points, axis=0)
        normalized = points - centroid
        scale = np.max(np.linalg.norm(normalized, axis=1))
        return normalized / scale if scale > 0 else normalized
    
    def compute_signed_area(polygon):
        """Вычисление ориентированной площади многоугольника"""
        x, y = polygon[:, 0], polygon[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    
    def compute_curvature_6_points(points):
        """Вычисление дискретной кривизны для 6 точек"""
        n = len(points)
        curvatures = []
        
        # Для внутренних точек (со 2-й по 5-ю) используем центральную разность
        for i in range(1, n-1):
            # Векторы до и после точки
            v_prev = points[i] - points[i-1]
            v_next = points[i+1] - points[i]
            
            # Углы
            norm_prev = np.linalg.norm(v_prev)
            norm_next = np.linalg.norm(v_next)
            
            if norm_prev > 0 and norm_next > 0:
                dot = np.dot(v_prev, v_next) / (norm_prev * norm_next)
                dot = np.clip(dot, -1.0, 1.0)
                angle = np.arccos(dot)
                
                # Направление (знак) кривизны
                cross = np.cross(v_prev, v_next)
                signed_angle = angle * np.sign(cross) if cross != 0 else angle
                
                # Кривизна как угол на единицу длины
                avg_length = (norm_prev + norm_next) / 2
                curvature = signed_angle / avg_length if avg_length > 0 else 0
                curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def compute_convex_hull_deviation(points, hull):
        """Вычисление отклонения от выпуклой оболочки"""
        hull_area = hull.volume if hasattr(hull, 'volume') else compute_signed_area(points[hull.vertices])
        curve_area = compute_signed_area(points)
        
        # Отношение площадей
        if abs(hull_area) > 1e-10:
            area_ratio = curve_area / hull_area
        else:
            area_ratio = 0
        
        # Среднее расстояние точек кривой до выпуклой оболочки
        deviations = []
        for i in range(len(points)):
            # Находим минимальное расстояние от точки i до любого ребра выпуклой оболочки
            min_dist = float('inf')
            
            # Проверяем, лежит ли точка на границе выпуклой оболочки
            if i in hull.vertices:
                continue
                
            for simplex in hull.simplices:
                # simplex - это пара индексов, определяющая ребро
                p1, p2 = points[simplex[0]], points[simplex[1]]
                
                # Вектор ребра
                edge_vec = p2 - p1
                point_vec = points[i] - p1
                
                # Проекция точки на ребро
                edge_length_sq = np.dot(edge_vec, edge_vec)
                if edge_length_sq > 0:
                    t = np.dot(point_vec, edge_vec) / edge_length_sq
                    t = max(0, min(1, t))  # Ограничиваем точкой на отрезке
                    
                    # Ближайшая точка на ребре
                    closest = p1 + t * edge_vec
                    dist = np.linalg.norm(points[i] - closest)
                    min_dist = min(min_dist, dist)
            
            if min_dist < float('inf'):
                deviations.append(min_dist)
        
        avg_deviation = np.mean(deviations) if deviations else 0
        
        return area_ratio, avg_deviation
    
    def compute_angle_metrics(points):
        """Вычисление угловых характеристик"""
        n = len(points)
        signed_angles = []
        total_turning = 0
        
        for i in range(n-2):
            v1 = points[i+1] - points[i]
            v2 = points[i+2] - points[i+1]
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                dot = np.dot(v1, v2) / (norm1 * norm2)
                dot = np.clip(dot, -1.0, 1.0)
                angle = np.arccos(dot)
                
                # Определяем знак угла
                cross = np.cross(v1, v2)
                signed_angle = angle * np.sign(cross) if cross != 0 else angle
                signed_angles.append(signed_angle)
                total_turning += signed_angle
        
        signed_angles = np.array(signed_angles)
        
        # Для замкнутой кривой добавляем последний угол
        if n >= 3:
            v_last = points[0] - points[-1]
            v_first = points[1] - points[0]
            
            norm_last = np.linalg.norm(v_last)
            norm_first = np.linalg.norm(v_first)
            
            if norm_last > 0 and norm_first > 0:
                dot = np.dot(v_last, v_first) / (norm_last * norm_first)
                dot = np.clip(dot, -1.0, 1.0)
                angle = np.arccos(dot)
                cross = np.cross(v_last, v_first)
                signed_angle = angle * np.sign(cross) if cross != 0 else angle
                signed_angles = np.append(signed_angles, signed_angle)
                total_turning += signed_angle
        
        return {
            'angles': signed_angles,
            'total_turning': total_turning,
            'mean_angle': np.mean(signed_angles) if len(signed_angles) > 0 else 0,
            'angle_variance': np.var(signed_angles) if len(signed_angles) > 0 else 0,
            'max_angle': np.max(np.abs(signed_angles)) if len(signed_angles) > 0 else 0
        }
    
    # Проверка входных данных
    if points.shape != (6, 2):
        raise ValueError(f"Ожидается массив формы (6, 2), получен {points.shape}")
    
    # Нормализуем точки
    norm_points = normalize_points(points)
    
    try:
        # 1. Метрика выпуклой оболочки
        hull = ConvexHull(norm_points)
        area_ratio, avg_deviation = compute_convex_hull_deviation(norm_points, hull)
        
        # Знак показывает выпуклость/вогнутость
        convexity_sign = np.sign(area_ratio) if abs(area_ratio) > 1e-10 else 0
        
        # 2. Метрика кривизны
        curvatures = compute_curvature_6_points(norm_points)
        
        # Статистики кривизны
        curvature_mean = np.mean(curvatures) if len(curvatures) > 0 else 0
        curvature_std = np.std(curvatures) if len(curvatures) > 0 else 0
        curvature_max = np.max(np.abs(curvatures)) if len(curvatures) > 0 else 0
        
        # 3. Угловые метрики
        angle_metrics = compute_angle_metrics(norm_points)
        
        # 4. Дополнительная метрика: отклонение от средней кривизны
        if len(curvatures) > 0:
            curvature_skew = np.mean(curvatures**3) / (curvature_std**3 + 1e-10)
        else:
            curvature_skew = 0
        
        # 5. Композитная метрика (настроена для 6 точек)
        weights = {
            'area_ratio': 0.35,
            'avg_deviation': 0.25,
            'curvature_mean': 0.20,
            'total_turning': 0.20
        }
        
        # Нормализация компонентов
        components = {
            'area_ratio': np.tanh(abs(area_ratio)) * convexity_sign,
            'avg_deviation': np.tanh(avg_deviation * 5) * (-convexity_sign if convexity_sign != 0 else 1),
            'curvature_mean': np.tanh(curvature_mean * 3),
            'total_turning': np.tanh(angle_metrics['total_turning'] * 0.5)
        }
        
        # Итоговая метрика
        metric = sum(weights[key] * components[key] for key in weights)
        
        # Метаданные для анализа
        metadata = {
            'components': components,
            'convexity_sign': convexity_sign,
            'area_ratio': area_ratio,
            'avg_deviation': avg_deviation,
            'curvature_stats': {
                'mean': curvature_mean,
                'std': curvature_std,
                'max': curvature_max,
                'skew': curvature_skew
            },
            'angle_stats': angle_metrics,
            'hull_area': hull.volume if hasattr(hull, 'volume') else compute_signed_area(norm_points[hull.vertices]),
            'curve_area': compute_signed_area(norm_points),
            'hull_vertices': hull.vertices.tolist()
        }
        
        return metric, metadata
        
    except Exception as e:
        # В случае ошибки возвращаем нулевую метрику
        print(f"Warning: {e}")
        return 0.0, {'error': str(e), 'components': {}, 'convexity_sign': 0}


def compare_curves_6_points(curve1: np.ndarray, curve2: np.ndarray) -> Dict:
    """
    Сравнивает две кривые (по 6 точек) по степени выпуклости/вогнутости.
    
    Возвращает:
    -----------
    Dict : Сравнительные характеристики
    """
    metric1, meta1 = curve_convexity_metric_6_points(curve1)
    metric2, meta2 = curve_convexity_metric_6_points(curve2)
    
    # Интерпретация значений
    def interpret_metric(value):
        abs_val = abs(value)
        sign = "выпуклая" if value > 0 else "вогнутая"
        
        if abs_val < 0.05:
            return "почти прямая"
        elif abs_val < 0.15:
            return f"очень слабо {sign}"
        elif abs_val < 0.3:
            return f"слабо {sign}"
        elif abs_val < 0.5:
            return f"умеренно {sign}"
        elif abs_val < 0.7:
            return f"сильно {sign}"
        else:
            return f"очень сильно {sign}"
    
    # Определение, какая кривая более выпуклая
    if abs(metric1) > abs(metric2):
        more_convex = 1 if metric1 > 0 else -1
        more_concave = 2 if metric2 < 0 else -1
    else:
        more_convex = 2 if metric2 > 0 else -1
        more_concave = 1 if metric1 < 0 else -1
    
    result = {
        'curve1': {
            'metric': metric1,
            'type': interpret_metric(metric1),
            'abs_metric': abs(metric1)
        },
        'curve2': {
            'metric': metric2,
            'type': interpret_metric(metric2),
            'abs_metric': abs(metric2)
        },
        'comparison': {
            'difference': abs(metric1 - metric2),
            'ratio': metric1 / metric2 if abs(metric2) > 1e-10 else float('inf'),
            'more_convex_curve': more_convex,
            'more_concave_curve': more_concave,
            'summary': f"Кривая 1: {interpret_metric(metric1)}, Кривая 2: {interpret_metric(metric2)}"
        },
        'metadata': {
            'curve1': meta1,
            'curve2': meta2
        }
    }
    
    # Добавляем текстовое сравнение
    if metric1 > metric2 and metric1 > 0:
        result['comparison']['text'] = "Первая кривая более выпуклая"
    elif metric2 > metric1 and metric2 > 0:
        result['comparison']['text'] = "Вторая кривая более выпуклая"
    elif metric1 < metric2 and metric1 < 0:
        result['comparison']['text'] = "Первая кривая более вогнутая"
    elif metric2 < metric1 and metric2 < 0:
        result['comparison']['text'] = "Вторая кривая более вогнутая"
    else:
        result['comparison']['text'] = "Кривые имеют схожую выпуклость/вогнутость"
    
    return result


# Дополнительные утилиты для анализа кривых
def generate_test_curves() -> Dict[str, np.ndarray]:
    """Генерация тестовых кривых с 6 точками"""
    
    # 1. Выпуклая кривая (парабола)
    convex = np.array([
        [0, 0],
        [1, 0.5],
        [2, 2],
        [3, 4.5],
        [4, 2],
        [5, 0]
    ])
    
    # 2. Вогнутая кривая
    concave = np.array([
        [0, 5],
        [1, 3.5],
        [2, 2],
        [3, 2],
        [4, 3.5],
        [5, 5]
    ])
    
    # 3. S-образная кривая
    s_shaped = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 2],
        [4, 1],
        [5, 0]
    ])
    
    # 4. Волнообразная кривая
    wavy = np.array([
        [0, 0],
        [1, 1],
        [2, 0],
        [3, -1],
        [4, 0],
        [5, 1]
    ])
    
    # 5. Почти прямая линия
    almost_straight = np.array([
        [0, 0],
        [1, 0.1],
        [2, 0],
        [3, -0.1],
        [4, 0],
        [5, 0.1]
    ])
    
    return {
        'convex': convex,
        'concave': concave,
        's_shaped': s_shaped,
        'wavy': wavy,
        'almost_straight': almost_straight
    }


def visualize_comparison(curves_dict: Dict[str, np.ndarray]):
    """Визуализация и сравнение нескольких кривых"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = {}
    
    for idx, (name, curve) in enumerate(curves_dict.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Вычисляем метрику
        metric, metadata = curve_convexity_metric_6_points(curve)
        metrics[name] = metric
        
        # Рисуем кривую
        ax.plot(curve[:, 0], curve[:, 1], 'bo-', linewidth=2, markersize=6)
        ax.scatter(curve[:, 0], curve[:, 1], color='red', s=50, zorder=5)
        
        # Добавляем выпуклую оболочку
        try:
            hull = ConvexHull(curve)
            for simplex in hull.simplices:
                ax.plot(curve[simplex, 0], curve[simplex, 1], 'r--', alpha=0.5)
        except:
            pass
        
        # Заголовок с информацией
        ax.set_title(f'{name}\nМетрика: {metric:.3f}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')
    
    # Удаляем лишние оси
    for idx in range(len(curves_dict), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()
    
    # Выводим сравнение
    print("\nСравнение кривых:")
    print("-" * 50)
    for name, metric in sorted(metrics.items(), key=lambda x: abs(x[1]), reverse=True):
        sign = "+" if metric > 0 else "-"
        print(f"{name:15} | {sign}{abs(metric):.4f} | {interpret_metric(metric)}")
    
    return metrics


def interpret_metric(value):
    """Интерпретация численной метрики"""
    abs_val = abs(value)
    sign = "выпуклая" if value > 0 else "вогнутая"
    
    if abs_val < 0.05:
        return "почти прямая"
    elif abs_val < 0.15:
        return f"очень слабо {sign}"
    elif abs_val < 0.3:
        return f"слабо {sign}"
    elif abs_val < 0.5:
        return f"умеренно {sign}"
    elif abs_val < 0.7:
        return f"сильно {sign}"
    else:
        return f"очень сильно {sign}"


# Пример использования
if __name__ == "__main__":
    # Создаем тестовые кривые
    curves = generate_test_curves()
    
    # Тестируем каждую кривую
    for name, curve in curves.items():
        print(f"\n{'='*50}")
        print(f"Анализ кривой: {name}")
        print(f"Точки:\n{curve}")
        
        metric, metadata = curve_convexity_metric_6_points(curve)
        
        print(f"\nМетрика выпуклости: {metric:.4f}")
        print(f"Интерпретация: {interpret_metric(metric)}")
        print(f"Знак: {'положительный (выпуклая)' if metric > 0 else 'отрицательный (вогнутая)'}")
        print(f"Абсолютное значение: {abs(metric):.4f}")
        
        if 'components' in metadata:
            print("\nКомпоненты метрики:")
            for comp_name, comp_value in metadata['components'].items():
                print(f"  {comp_name}: {comp_value:.4f}")
    
    # Сравниваем две конкретные кривые
    print(f"\n{'='*50}")
    print("Сравнение выпуклой и вогнутой кривых:")
    
    comparison = compare_curves_6_points(curves['convex'], curves['concave'])
    
    print(f"\nКривая 1 (выпуклая):")
    print(f"  Метрика: {comparison['curve1']['metric']:.4f}")
    print(f"  Тип: {comparison['curve1']['type']}")
    
    print(f"\nКривая 2 (вогнутая):")
    print(f"  Метрика: {comparison['curve2']['metric']:.4f}")
    print(f"  Тип: {comparison['curve2']['type']}")
    
    print(f"\nСравнение:")
    print(f"  {comparison['comparison']['text']}")
    print(f"  Разница: {comparison['comparison']['difference']:.4f}")
    
    # Визуализация (раскомментировать, если установлен matplotlib)
    # visualize_comparison(curves)
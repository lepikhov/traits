import math
from collections import Counter
from traits_utils import draw_points_and_lines
import cv2
import numpy as np

DEBUG = True
#DEBUG = False


def debug_print(*args):
    if DEBUG:
        print(*args)

def make_vector(p1, p2): #вектор из точек
    return p2[0] - p1[0], p2[1] - p1[1]

def get_len(v): #длина вектора
    return math.sqrt(v[0] ** 2 + v[1] ** 2)

def trait_from_two_vectors(v1, v2, coef1=1.0, coef2=0.0, coef3=0.0):
    v1_len = get_len(v1)
    v2_len = get_len(v2)

    debug_print(coef1, coef2, coef3)
    debug_print(v1_len, v2_len)

    if (v1_len < coef1*v2_len*(1-coef2)):
        trait = 3
    elif (v1_len > coef1*v2_len*(1+coef3)):
        trait = 1
    else:
        trait = 2     
    return trait

def trait_from_angle(p1, p2, p3, p4, threshold, adjacent=False, coef1=0.0, coef2=0.0):
    try: 
        angl = abs(angle(p1, p2, p3, p4))
    except:
        angl = 0
    
    if adjacent and angl > 90:
        angl = 180 - angl
    debug_print(angl)
    
    if (angl < threshold*(1-coef1)):
        trait = 3
    elif (angl > threshold*(1+coef2)):
        trait = 1
    else:
        trait = 2     
    return trait    
    
            

def check_keypoints(keypoints, check_list):
    if max(check_list) >= len(keypoints):
        return False

    for k in check_list:
        if not keypoints[k][2]:
            return False

    return True        

def get_points(keypoints, points_list):
    points = []

    if not check_keypoints(keypoints, points_list):
        return points
            
    for p in points_list:
        points.append((keypoints[p][0], keypoints[p][1], p+1))

    return points        

def get_lines(keypoints, lines_list):
    lines = []

    #if not check_keypoints:
    #    return lines
            
    for l in lines_list:
        lines.append(((keypoints[l[0]][0], keypoints[l[0]][1]), (keypoints[l[1]][0], keypoints[l[1]][1])))

    return lines        

def detect_horizon(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Детекция краев (Canny)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Поиск линий с помощью преобразования Хафа
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                           minLineLength=100, maxLineGap=10)
    #print(lines)    
    
    # Фильтрация горизонтальных линий
    horizon_candidates = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180/np.pi)
            
            # Отбираем почти горизонтальные линии (угол близкий к 0 или 180)
            if angle < 20 or angle > 160:
                horizon_candidates.append(((x1 + x2) / 2, (y1 + y2) / 2))
    
    print(horizon_candidates)
    if horizon_candidates:
        horizon = np.median(horizon_candidates)
        return horizon
    
    return None

def detect_horizon_line(image, return_type="slope_intercept"):
    """
    Определяет линию горизонта, включая наклонные случаи
    
    Параметры:
    - image: изображение cv2
    - return_type: тип возвращаемых данных
        "slope_intercept": возвращает (угловой_коэффициент, смещение)
        "points": возвращает две точки для отрисовки линии
        "angle": возвращает угол наклона в градусах
    
    Возвращает:
    - В зависимости от return_type: параметры линии, точки или угол
    - None если линию не удалось определить
    """

        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Улучшаем детекцию краев
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Убираем шум
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Поиск линий с помощью преобразования Хафа
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                           minLineLength=100, maxLineGap=20)
    
    if lines is None:
        return None
    
    # Фильтрация и выбор лучшей линии горизонта
    horizon_candidates = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Игнорируем почти вертикальные линии
        if abs(x2 - x1) < 10:  # предотвращаем деление на ноль
            continue
            
        # Вычисляем угол наклона
        slope = (y2 - y1) / (x2 - x1)
        angle = np.degrees(np.arctan(slope))
        
        # Фильтруем по углу (исключаем слишком крутые наклоны)
        if abs(angle) < 45:  # горизонт обычно не слишком крутой
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            horizon_candidates.append({
                'points': (x1, y1, x2, y2),
                'slope': slope,
                'intercept': y1 - slope * x1,
                'angle': angle,
                'length': length
            })
    
    if not horizon_candidates:
        return None
    
    # Выбираем самую длинную линию как горизонт
    best_line = max(horizon_candidates, key=lambda x: x['length'])
    
    # Возвращаем результат в нужном формате
    if return_type == "slope_intercept":
        return best_line['slope'], best_line['intercept']
    
    elif return_type == "points":
        x1, y1, x2, y2 = best_line['points']
        # Расширяем линию на всю ширину изображения
        height, width = image.shape[:2]
        if abs(best_line['slope']) > 0.001:  # не горизонтальная линия
            y1_extended = int(best_line['slope'] * 0 + best_line['intercept'])
            y2_extended = int(best_line['slope'] * width + best_line['intercept'])
            return (0, y1_extended), (width, y2_extended)
        else:  # горизонтальная линия
            return (0, int(best_line['intercept'])), (width, int(best_line['intercept']))
    
    elif return_type == "angle":
        return best_line['angle']
    
    return best_line


def get_horizont_line(keypoints):
    
    
    def find_closest_line(points):
        # Шаг 1: Вычисление среднего
        mean_x = sum(p[0] for p in points) / len(points)
        mean_y = sum(p[1] for p in points) / len(points)
        
        # Шаг 2: Центрирование точек и вычисление матрицы ковариации
        Cxx, Cxy, Cyy = 0, 0, 0
        for x, y in points:
            dx = x - mean_x
            dy = y - mean_y
            Cxx += dx * dx
            Cxy += dx * dy
            Cyy += dy * dy
        
        # Шаг 3: Вычисление собственного значения и вектора
        discriminant = math.sqrt((Cxx - Cyy)**2 + 4 * Cxy**2)
        lambda1 = (Cxx + Cyy + discriminant) / 2
        
        # Шаг 4: Выбор собственного вектора
        if abs(Cxy) > 1e-12:
            vx = Cxy
            vy = lambda1 - Cxx
        else:
            if Cxx >= Cyy:
                vx, vy = 1, 0
            else:
                vx, vy = 0, 1
        
        # Шаг 5: Нормирование вектора
        length = math.sqrt(vx**2 + vy**2)
        if length < 1e-12:
            vx, vy = 1, 0
        else:
            vx /= length
            vy /= length
        
        # Шаг 6: Составление уравнения прямой
        A = vy
        B = -vx
        C = vx * mean_y - vy * mean_x
    
        return A, B, C
    
    if not check_keypoints(keypoints, [38, 39, 62, 63]):
        return (0,0), (100, 0)
    
    A,B,C = find_closest_line(((keypoints[38][0], keypoints[38][1]),
                               (keypoints[62][0], keypoints[62][1]),
                               (keypoints[39][0], keypoints[39][1]),
                               (keypoints[63][0], keypoints[63][1]),
    ))
    
    debug_print(A, B, C)
    if abs(B) < 1e-12:
        p1 = [-C/(A+1e-12), 0]
        p2 = [-C/(A+1e-12), 100]
    else:
        p1 = (keypoints[38][0], -(C+A*keypoints[38][0])/B)
        p2 = (keypoints[62][0], -(C+A*keypoints[62][0])/B) 
        
    return p1, p2              
        
        
def distance_point_to_line(p1, p2, p3):
    """
    Вычисляет расстояние от точки p3 до прямой, проходящей через точки p1 и p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Если точки p1 и p2 совпадают, возвращаем расстояние между p1 и p3
    if math.isclose(x1, x2) and math.isclose(y1, y2):
        return math.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    
    # Вычисляем числитель (модуль векторного произведения)
    numerator = abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
    # Вычисляем знаменатель (длину вектора прямой)
    denominator = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return numerator / denominator    

def find_parallel(p1, p2, p3):
    # Вычисляем вектор от p1 к p2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Получаем p4, прибавляя вектор к p3
    p4 = (p3[0] + dx, p3[1] + dy)
    # Получаем p5, вычитая вектор из p3
    p5 = (p3[0] - dx, p3[1] - dy)
        
    return p4, p5

def angle(A, B, C, D):
    '''Получаем угол между отрезками AB и CD '''
  
    def get_angle(v1, v2): #угол между двумя векторами
        a = math.acos((v1[0] * v2[0] + v1[1] * v2[1]) / (get_len(v1) * get_len(v2)))
        angle = math.degrees(a)
        return angle
    
    if len(A) == len(B) == len(C) == len(D) == 2:
        ab = make_vector(A, B)
        cd = make_vector(C ,D)
        angle_a = get_angle(ab, cd)
        return angle_a
    else: 
        raise ValueError
        

def calculate_traits(keypoints, breed='any', draw=False, image=None):
    
    traits = {}

    horizont_line = get_horizont_line(keypoints)
    debug_print(horizont_line)
    
    horizont_line_cv2 = detect_horizon_line(image, return_type="points")
    debug_print('horizont:', horizont_line_cv2)
    
    #head_0
    debug_print('head_0')
    if not check_keypoints(keypoints, [0, 7, 18, 19, 45, 65]):

        traits['head_0'] = 0

    else:
        
        neck_k = 0.02
        body_k = 0.02
        rump_k = 0.02
        
        v_head = make_vector([keypoints[0][0],keypoints[0][1]], [keypoints[7][0],keypoints[7][1]])
        v_neck = make_vector([keypoints[7][0],keypoints[7][1]], [keypoints[65][0],keypoints[65][1]])
        v_body = make_vector([keypoints[45][0],keypoints[45][1]], [keypoints[18][0],keypoints[18][1]])
        v_rump = make_vector([keypoints[19][0],keypoints[19][1]], [keypoints[18][0],keypoints[18][1]])
             
        debug_print(v_head, v_neck, v_body, v_rump)
             
        head_neck = trait_from_two_vectors(v_head, v_neck, 1.0, neck_k, neck_k)
        head_body = trait_from_two_vectors(v_head, v_body, (1/3), body_k, body_k)  
        head_rump = trait_from_two_vectors(v_head, v_rump, 1.0, rump_k, rump_k)              
                                       

        debug_print(head_neck, head_body, head_rump)                
        traits['head_0'] = Counter((head_neck, head_body, head_rump)).most_common(1)[0][0]

        if draw:
            points = get_points(keypoints, [0, 7, 18, 19, 45, 65])
            lines = get_lines(keypoints, [(0,7), (7, 65), (45, 18), (19, 18)])
            draw_points_and_lines(points=points,lines=lines,image=image,path='./outputs/__calculate_head_0.png')

    
    #nape
    debug_print('nape')
    if not check_keypoints(keypoints, [4, 5, 7, 8]):

        traits['nape'] = 0

    else:
        
        forehead_k = 0.02
        
        v_nape = make_vector([keypoints[7][0],keypoints[7][1]], [keypoints[8][0],keypoints[8][1]])
        v_forehead = make_vector([keypoints[4][0],keypoints[4][1]], [keypoints[5][0],keypoints[5][1]])    
        
        debug_print(v_nape, v_forehead)
           
        traits['nape'] = trait_from_two_vectors(v_nape, v_forehead, (1/2), forehead_k, forehead_k)

        if draw:
            points = get_points(keypoints, [4, 5, 7, 8])
            lines = get_lines(keypoints, [(7,8), (4, 5)])
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_nape.png')

    #neck_0
    debug_print('neck_0')
    if not check_keypoints(keypoints, [7, 18, 45, 65]):

        traits['neck_0'] = 0

    else:
        
        body_k = 0.02
        
        v_neck = make_vector([keypoints[7][0],keypoints[7][1]], [keypoints[65][0],keypoints[65][1]])
        v_body = make_vector([keypoints[18][0],keypoints[18][1]], [keypoints[45][0],keypoints[45][1]])    
        
        debug_print(v_neck, v_body)
           
        traits['neck_0'] = trait_from_two_vectors(v_neck, v_body, (1/2), body_k, body_k)     

        if draw:
            points = get_points(keypoints, [7, 18, 45, 65])
            lines = get_lines(keypoints, [(7,65), (18, 45)])
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_neck_0.png')           

    #'angle_4'     
    debug_print('angle_4')     
    if not check_keypoints(keypoints, [6, 45, 63, 64]):

        traits['angle_4'] = 0

    else:
        
        angle_4_k = 0.02

        v_horse = make_vector([keypoints[6][0],keypoints[6][1]], [keypoints[63][0],keypoints[63][1]])
        diff_y = (keypoints[64][1]-keypoints[45][1])/(get_len(v_horse)+0.001)

        debug_print(diff_y)

        if (diff_y > angle_4_k):
            traits['angle_4'] = 1
        elif (diff_y < -angle_4_k):  
            traits['angle_4'] = 3
        else:
            traits['angle_4'] = 2       

        if draw:
            points = get_points(keypoints, [6, 45, 63, 64])
            lines = get_lines(keypoints, [(6, 63), (64, 45)])
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_angle_4.png')                                              
    
    #withers_0
    debug_print('withers_0')     
    traits['withers_0'] = 0

    #withers_1
    debug_print('withers_1')  

    if not check_keypoints(keypoints, [11, 12]):
   
        traits['withers_1'] = 0            

    else:
        withers_height = distance_point_to_line(horizont_line[0], horizont_line[1], (keypoints[11][0], keypoints[11][1]))
        first_spinal_vertebra_height = distance_point_to_line(horizont_line[0], horizont_line[1], (keypoints[12][0], keypoints[12][1]))
        
        debug_print(withers_height, first_spinal_vertebra_height)
        
        ratio = 200.0*(withers_height - first_spinal_vertebra_height)/(first_spinal_vertebra_height + withers_height + 1e-6)
        
        debug_print(ratio)
        
        match breed:
            case 'verkhovaya':
                low_withers_threshold = 8.0
                high_withers_threshold = 10.0
            case 'tyazhelovoz':
                low_withers_threshold = 4.0
                high_withers_threshold = 7.0   
            case 'orlovskaya':
                low_withers_threshold = 6.0
                high_withers_threshold = 8.0  
            case _:               
                low_withers_threshold = 6.0
                high_withers_threshold = 8.0                  

        if ratio > high_withers_threshold:
            traits['withers_1'] = 1
        elif ratio < low_withers_threshold:
            traits['withers_1'] = 2
        else:
            traits['withers_1'] = 2                                   
        
        if draw:
            points = get_points(keypoints, [11, 12])
            lines = [horizont_line]
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_withers_1.png')                                              
    
        

    #shoulder
    debug_print('shoulder')
    if not check_keypoints(keypoints, [11, 45, 48]):

        traits['shoulder'] = 0

    else:        
        shoulder_k = 0.02
        
        v_shoulder = make_vector([keypoints[11][0],keypoints[11][1]], [keypoints[45][0],keypoints[45][1]])
        v_perpendicular = make_vector([keypoints[45][0],keypoints[45][1]], [keypoints[48][0],keypoints[48][1]])    
        
        debug_print(v_shoulder, v_perpendicular)
           
        traits['shoulder'] = trait_from_two_vectors(v_shoulder, v_perpendicular, 1.0, shoulder_k, shoulder_k)
        
        if draw:
            points = get_points(keypoints, [11, 45, 48])
            lines = get_lines(keypoints, [(11, 45), (45, 48)])
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_shoulders.png')          


    #'angle_5'     
    debug_print('angle_5')     
    if not check_keypoints(keypoints, [11, 45]):

        traits['angle_5'] = 0

    else:                     
        
        angle_5_k = 0.02
        
        traits['angle_5']=trait_from_angle(horizont_line[0], horizont_line[1], 
                                           (keypoints[11][0],keypoints[11][1]), 
                                           (keypoints[45][0],keypoints[45][1]),
                                           threshold=45, coef1=angle_5_k, coef2=angle_5_k)
        
        if draw:
            points = get_points(keypoints, [11, 45])
            lines = get_lines(keypoints, [(11, 45)])
            lines.append(horizont_line)
            aux_line = find_parallel(horizont_line[0], horizont_line[1], (keypoints[11][0], keypoints[11][1]))  
            lines.append(aux_line)
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_angle_5.png')   
    
    #lower_back_0
    debug_print('lower_back_0')     
    traits['lower_back_0'] = 0
    
    #'spine_0'
    debug_print('spine_0')
    if not check_keypoints(keypoints, [12, 14, 45, 71]):

        traits['spine_0'] = 0

    else:      
    
        spine_0_k = 0.02
        
        v_spine = make_vector([keypoints[12][0],keypoints[12][1]], [keypoints[14][0],keypoints[14][1]])
        v_chest = make_vector([keypoints[45][0],keypoints[45][1]], [keypoints[71][0],keypoints[71][1]])    
        
        debug_print(v_spine, v_chest)
           
        traits['spine_0'] = trait_from_two_vectors(v_chest, v_spine, (1/3), spine_0_k, spine_0_k)    
        
        if draw:
            points = get_points(keypoints, [12, 14, 45, 71])
            lines = get_lines(keypoints, [(12, 14), (45, 71)])
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_spine_0.png')           
                
                
    #spine_3
    debug_print('spine_3')     
    traits['spine_3'] = 0
            
    #'rump'
    debug_print('rump')
    if not check_keypoints(keypoints, [18, 19, 45]):

        traits['rump'] = 0

    else:      
    
        rump_k_1 = 0.0357
        rump_k_2 = 0.0344
        
        v_rump = make_vector([keypoints[18][0],keypoints[18][1]], [keypoints[19][0],keypoints[19][1]])
        v_oblique_body_length = make_vector([keypoints[18][0],keypoints[18][1]], [keypoints[45][0],keypoints[45][1]])    
        
        debug_print(v_rump, v_oblique_body_length)
           
        traits['rump'] = trait_from_two_vectors(v_rump, v_oblique_body_length, (0.29), rump_k_1, rump_k_2)    
        
        if draw:
            points = get_points(keypoints, [18, 19, 45])
            lines = get_lines(keypoints, [(18, 19), (18, 45)])
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_rump.png')           
        
    #'angle_10'     
    debug_print('angle_10')     
    if not check_keypoints(keypoints, [16, 17]):

        traits['angle_10'] = 0

    else:                     
        
        angle_10_k_1 = 0.25
        angle_10_k_2 = 0.16
        
        traits['angle_10']=trait_from_angle(horizont_line[0], horizont_line[1], 
                                           (keypoints[16][0],keypoints[16][1]), 
                                           (keypoints[17][0],keypoints[17][1]),
                                           threshold=25, adjacent=True, coef1=angle_10_k_1, coef2=angle_10_k_2)    
        
        if draw:
            points = get_points(keypoints, [16, 17])
            lines = get_lines(keypoints, [(16, 17)])
            lines.append(horizont_line)
            aux_line = find_parallel(horizont_line[0], horizont_line[1], (keypoints[16][0], keypoints[16][1]))  
            lines.append(aux_line)
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_angle_10.png')                     
    
    #rib_cage_0
    debug_print('rib_cage_0')
    if not check_keypoints(keypoints, [6, 43, 47, 63]):

        traits['rib_cage_0'] = 0

    else:   
        rib_cage_k = 0.03

        v_horse = make_vector([keypoints[6][0],keypoints[6][1]], [keypoints[63][0],keypoints[63][1]])
        diff_y = (keypoints[47][1]-keypoints[43][1])/(get_len(v_horse)+0.001)

        debug_print(diff_y)

        if (diff_y > rib_cage_k):
            traits['rib_cage_0'] = 1
        elif (diff_y < -rib_cage_k):  
            traits['rib_cage_0'] = 3
        else:
            traits['rib_cage_0'] = 2   
            
        if draw:
            points = get_points(keypoints, [43, 47])
            lines = []
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_rib_cage_0.png')               
    
    #falserib_0
    debug_print('falserib_0')
    if not check_keypoints(keypoints, [14, 16, 20, 40]):

        traits['falserib_0'] = 0

    else:      
    
        falserib_k = 0.02
        
        v_1 = make_vector([keypoints[14][0],keypoints[14][1]], [keypoints[40][0],keypoints[40][1]])
        v_2 = make_vector([keypoints[16][0],keypoints[16][1]], [keypoints[20][0],keypoints[20][1]])    
        
        debug_print(v_1, v_2)
           
        traits['falserib_0'] = trait_from_two_vectors(v_1, v_2, 1.0, falserib_k, falserib_k)    
        
        if draw:
            points = get_points(keypoints, [14, 16, 20, 40])
            lines = get_lines(keypoints, [(14, 40), (16, 20)])
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_falserib_0.png')                 
        
    #forearm
    debug_print('forearm')
    if not check_keypoints(keypoints, [46, 48, 55]):

        traits['forearm'] = 0

    else:      
    
        forearm_k = 0.02
        
        v_forearm = make_vector([keypoints[46][0],keypoints[46][1]], [keypoints[48][0],keypoints[48][1]])
        v_metacarpus = make_vector([keypoints[48][0],keypoints[48][1]], [keypoints[55][0],keypoints[55][1]])    
        
        debug_print(v_forearm, v_metacarpus)
           
        traits['forearm'] = trait_from_two_vectors(v_forearm, v_metacarpus, (1/3), forearm_k, forearm_k)     
        
        if draw:
            points = get_points(keypoints, [46, 48, 55])
            lines = get_lines(keypoints, [(46, 48), (48, 55)])
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_forearm.png')               
        
    #headstock
    debug_print('headstock')
    if not check_keypoints(keypoints, [48, 52, 55]):

        traits['headstock'] = 0

    else:      
    
        headstock_k = 0.02
        
        v_headstock = make_vector([keypoints[52][0],keypoints[52][1]], [keypoints[55][0],keypoints[55][1]])
        v_metacarpus = make_vector([keypoints[48][0],keypoints[48][1]], [keypoints[55][0],keypoints[55][1]])    
        
        debug_print(v_headstock, v_metacarpus)
           
        traits['headstock'] = trait_from_two_vectors(v_headstock, v_metacarpus, (1/3), headstock_k, headstock_k)  
        
        if draw:
            points = get_points(keypoints, [48, 52, 55])
            lines = get_lines(keypoints, [(52, 55), (48, 55)])
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_headstock.png')          
                    
    
    #'angle_12'     
    debug_print('angle_12')     
    if not check_keypoints(keypoints, [55, 63]):

        traits['angle_12'] = 0

    else:                     
        
        angle_12_k_1 = 0.0178
        angle_12_k_2 = 0.0178
        
        traits['angle_12'] = trait_from_angle(horizont_line[0], horizont_line[1], 
                                           (keypoints[63][0],keypoints[63][1]), 
                                           (keypoints[55][0],keypoints[55][1]),
                                           threshold=56, adjacent=False, coef1=angle_12_k_1, coef2=angle_12_k_2)    
        
        if draw:
            points = get_points(keypoints, [55, 63])
            lines = get_lines(keypoints, [(55, 63)])
            lines.append(horizont_line)
            aux_line = find_parallel(horizont_line[0], horizont_line[1], (keypoints[63][0], keypoints[63][1]))  
            lines.append(aux_line)
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_angle_12.png')    
        
    #shin_0
    debug_print('shin_0')
    if not check_keypoints(keypoints, [20, 24, 31]):

        traits['shin_0'] = 0

    else:      
    
        shin_0_k = 0.02
        
        v_shin_0 = make_vector([keypoints[20][0],keypoints[20][1]], [keypoints[24][0],keypoints[24][1]])
        v_metatarsus = make_vector([keypoints[24][0],keypoints[24][1]], [keypoints[31][0],keypoints[31][1]])    
        
        debug_print(v_shin_0, v_metatarsus)
           
        traits['shin_0'] = trait_from_two_vectors(v_shin_0, v_metatarsus, (3/2), shin_0_k, shin_0_k)     
        
        if draw:
            points = get_points(keypoints, [20, 24, 31])
            lines = get_lines(keypoints, [(20, 24), (24, 31)])
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_shin_0.png')           
        
    #tailstock
    debug_print('tailstock')
    if not check_keypoints(keypoints, [24, 31, 38]):

        traits['tailstock'] = 0

    else:      
    
        tailstock_k = 0.02
        
        v_tailstock = make_vector([keypoints[31][0],keypoints[31][1]], [keypoints[38][0],keypoints[39][1]])
        v_metatarsus = make_vector([keypoints[24][0],keypoints[24][1]], [keypoints[31][0],keypoints[31][1]])    
        
        debug_print(v_tailstock, v_metatarsus)
           
        traits['tailstock'] = trait_from_two_vectors(v_tailstock, v_metatarsus, (1/3), tailstock_k, tailstock_k)  
        
        if draw:
            points = get_points(keypoints, [24, 31, 38])
            lines = get_lines(keypoints, [(31, 38), (24, 31)])
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_tailstock.png')                       


    #'angle_15'     
    debug_print('angle_15')     
    if not check_keypoints(keypoints, [31, 33]):

        traits['angle_15'] = 0

    else:                     
        
        angle_15_k_1 = 0.04
        angle_15_k_2 = 0.04
        
        traits['angle_15'] = trait_from_angle(horizont_line[0], horizont_line[1], 
                                           (keypoints[33][0],keypoints[33][1]), 
                                           (keypoints[31][0],keypoints[31][1]),
                                           threshold=62.5, adjacent=False, coef1=angle_15_k_1, coef2=angle_15_k_2)   
        
        if draw:
            points = get_points(keypoints, [31, 33])
            lines = get_lines(keypoints, [(31, 33)])
            lines.append(horizont_line)
            aux_line = find_parallel(horizont_line[0], horizont_line[1], (keypoints[33][0], keypoints[33][1]))  
            lines.append(aux_line)
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_angle_15.png')            
        
    #'angle_13'     
    debug_print('angle_13')     
    if not check_keypoints(keypoints, [20, 24, 31]):

        traits['angle_13'] = 0

    else:                     
        
        angle_13_k_1 = 0.066
        angle_13_k_2 = 0.066
        
        traits['angle_13'] = trait_from_angle((keypoints[24][0],keypoints[24][1]), 
                                           (keypoints[20][0],keypoints[20][1]), 
                                           (keypoints[24][0],keypoints[24][1]), 
                                           (keypoints[31][0],keypoints[31][1]),
                                           threshold=150, adjacent=False, coef1=angle_13_k_1, coef2=angle_13_k_2)      
        
        if draw:
            points = get_points(keypoints, [20, 24, 31])
            lines = get_lines(keypoints, [(20, 24), (24, 31)])
            draw_points_and_lines(points=points,lines=lines, image=image, path='./outputs/__calculate_angle_13.png')                     
            
    return traits
                
#--------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
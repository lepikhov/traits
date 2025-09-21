#ROOT_DATA_DIRECTORY ="/home/pavel/projects/horses/soft/python/morphometry/datasets/2024/traits" #path to dataset
ROOT_DATA_DIRECTORY ="/home/pavel/projects/horses/soft/python/morphometry/datasets/2025/new_photo_by_kalinkina" #path to dataset
ROOT_DATA_DIRECTORY_ORLOVSKAYA ="/home/pavel/projects/horses/soft/python/morphometry/datasets/2025/orlovskaya" #path to dataset

TPS_ENCODING = "utf8"
ALTERNATIVE_TPS_ENCODING = "cp1251"

#TRAITS_KEYS = ['head_0', 'nape', 'neck_0', 'withers_0', 'shoulder', 
#               'spine_0', 'rump', 'falserib_0', 'forearm', 'headstock', 
#               'hip', 'shin_0', 'tailstock', 'withers_1', 'spine_3', 
#               'lower_back_0', 'rib_cage_0', 'angle_4', 'angle_5', 
#               'angle_10', 'angle_11', 'angle_12', 'angle_13', 
#               'angle_14', 'angle_15'


TRAITS_KEYS = ['head_0', 'nape', 'neck_0', 'withers_0', 'shoulder', 
               'spine_0', 'rump', 'falserib_0', 'forearm', 'headstock', 
               'shin_0', 'tailstock', 'withers_1', 'spine_3', 
               'lower_back_0', 'rib_cage_0', 'angle_4', 'angle_5', 
               'angle_10', 'angle_12', 'angle_13', 
               'angle_15'
            ]

TRAITS_KEYS_MAP = {
    'head_0':("голова", ['не известно', 'большая', 'средняя', 'малая',]), 
    'nape':("затылок", ['не известно', 'длинный', 'средний', 'короткий',]),
    'neck_0':("шея (длина)", ['не известно', 'длинная', 'средняя', 'короткая',]),
    'withers_0':("холка, длина", ['не известно', 'длинная', 'средняя', 'короткая',]),
    'shoulder':("лопатка (длина)", ['не известно', 'длинная', 'средняя',  'короткая']),
    'spine_0':("спина (длина)", ['не известно', 'длинная', 'средняя', 'короткая',]),
    'rump':("круп(длина)", ['не известно', 'длинный',  'средний', 'короткий', ]),
    'falserib_0':("ложные ребра", ['не известно', 'длинные', 'средние', 'короткие',]),
    'forearm':("предплечье", ['не известно', 'длинное',  'среднее', 'короткое',]),
    'headstock':("передняя бабка(длина)", ['не известно', 'длинная', 'средняя', 'короткая',]),
    'hip':("бедро", ['не известно',  'длинное', 'среднее', 'короткое',]),
    'shin_0':("голень", ['не известно', 'длинная',  'средняя', 'короткая', ]),
    'tailstock':("задняя бабка (длина)", ['не известно', 'длинная', 'средняя', 'короткая',]),
    'withers_1':("холка (высота)", ['не известно', 'высокая', 'средняя', 'низкая',]),
    'spine_3':("спина", ['не известно', 'мягкая', 'прямая', 'выпуклая', ]),
    'lower_back_0':("поясница", ['не известно', 'запавшая', 'ровная', 'выпуклая']),
    'rib_cage_0':("грудная клетка", ['не известно', 'глубокая', 'средняя', 'не глубокая']),
    'angle_4':("шея выход", ['не известно', 'высокий', 'средний', 'низкий']),
    'angle_5':("лопатка", ['не известно', 'косая', 'средняя', 'прямая']),
    'angle_10':("круп", ['не известно', 'прямой', 'нормальный', 'свислый']),
    'angle_11':("передняя бабка (угол 11)", ['не известно', 'мягкая', 'нормальная', 'торцовая']),
    'angle_12':("передняя бабка (угол 12)", ['не известно', 'мягкая', 'нормальная',  'торцовая']),
    'angle_13':("скакательный сустав", ['не известно', 'саблистый', 'нормальный', 'прямой']),
    'angle_14':("задняя бабка (угол 14)", ['не известно', 'мягкая', 'нормальная', 'торцовая']),
    'angle_15':("задняя бабка (угол 15)", ['не известно', 'мягкая','нормальная', 'торцовая']),
}

"""
'head_0':{'не известно', 'большая', 'средняя', 'малая',}, 
'nape':{'не известно', 'длинный', 'средний', 'короткий',},
'neck_0':{'не известно', 'длинная', 'средняя', 'короткая',},
'withers_0':{'не известно', 'длинная', 'средняя', 'короткая',},
'shoulder':{'не известно', 'длинная', 'средняя',  'короткая'},
'spine_0':{'не известно', 'длинная', 'средняя', 'короткая',},
'rump':{'не известно', 'длинный',  'средний', 'короткий', },
'falserib_0':{'не известно', 'длинные', 'средние', 'короткие',},
'forearm':{'не известно', 'длинная',  'средняя', 'короткая',},
'headstock':{'не известно', 'длинная', 'средняя', 'короткая',},
'hip':{'не известно',  'длинная', 'средняя', 'короткая',},
'shin_0':{'не известно', 'длинная',  'средняя', 'короткая', },
'tailstock':{'не известно', 'длинная', 'средняя', 'короткая',},
'withers_1':{'не известно', 'высокая', 'средняя', 'низкая',},
'spine_3':{'не известно', 'мягкая', 'прямая', 'выпуклая', },
'lower_back_0':{'не известно', 'запавшая', 'ровная', 'выпуклая'},
'rib_cage_0':{'не известно', 'глубокая', 'средняя', 'не глубокая'},
'angle_4':{'не известно', 'высокий', 'средний', 'низкий'},
'angle_5':{'не известно', 'косая', 'средняя', 'прямая'},
'angle_10':{'не известно', 'прямой', 'нормальный', 'свислый'},
'angle_11':{'не известно', 'мягкие', 'нормальные', 'торцовые'},
'angle_12':{'не известно', 'мягкие', 'нормальные',  'торцовые'},
'angle_13':{'не известно', 'саблистый', 'нормальный', 'прямой'},
'angle_14':{'не известно', 'мягкие', 'нормальные', 'торцовые'},
'angle_15':{'не известно', 'мягкие','нормальные', 'торцовые'},
"""

TRAITS_KEYS_AUX = ['type', 'lower_back_len',]

#TRAITS_KEYS_AUX = ['type', 'lower_back_len',
#                   'head_0', 'neck_0', 'withers_0', 'shoulder', 
#               'spine_0', 'rump', 'falserib_0', 'forearm', 'headstock', 
#               'hip', 'shin_0', 'tailstock', 'withers_1', 'spine_3', 
#               'lower_back_0', 'rib_cage_0', 'angle_4', 'angle_5', 
#               'angle_10', 'angle_11', 'angle_12', 'angle_13', 
#               'angle_14', 'angle_15'
#                   ]

TRAITS_KEYS_SERVICE = ['ID', 'img_name',]

DEVICE = 'cuda'

models_weights=[
    '/home/pavel/projects/horses/soft/python/morphometry/traits/checkpoints/mobilenet-2024-09-08_19-55/checkpoint-000200.pth',
    '/home/pavel/projects/horses/soft/python/morphometry/traits/checkpoints/resnet-2024-09-08_21-22/checkpoint-000150.pth',   
    '/home/pavel/projects/horses/soft/python/morphometry/traits/checkpoints/squeezenet-2024-09-17_23-29/checkpoint-000200.pth',     
    '/home/pavel/projects/horses/soft/python/morphometry/traits/checkpoints/efficientnet-2024-09-18_23-35/checkpoint-000200.pth',
    '/home/pavel/projects/horses/soft/python/morphometry/traits/checkpoints/harmonicnet-2024-09-15_23-40/checkpoint-000200.pth',          
    '/home/pavel/projects/horses/soft/python/morphometry/traits/checkpoints/vitnet-2024-09-20_01-08/checkpoint-000100.pth',   
]


#Traits for different segments

SEGMENTATION_DIRECTORY = './outputs/segmentation/prepare_segments'

TRAITS_HEAD_NECK_KEYS = ['nape', ]

TRAITS_HEAD_NECK_BODY_KEYS = ['head_0', 'withers_0', 'spine_0', 'withers_1', 'rib_cage_0', 'angle_4', ]

#TRAITS_REAR_LEG_KEYS = ['hip', 'shin_0', 'tailstock',  'angle_13', 'angle_14', 'angle_15', ]
#TRAITS_REAR_LEG_KEYS = ['shin_0', 'tailstock',  'angle_13', 'angle_15', ]
TRAITS_REAR_LEG_KEYS = ['angle_15', ]

#TRAITS_FRONT_LEG_KEYS = [ 'headstock', 'angle_11', 'angle_12', ]
TRAITS_FRONT_LEG_KEYS = [ 'headstock', 'angle_12', ]

TRAITS_BODY_KEYS = ['rump', 'spine_3', 'lower_back_0',  'angle_10', ]

TRAITS_BODY_FRONT_LEG_KEYS = ['shoulder', 'falserib_0', 'forearm', 'angle_5', ]

TRAITS_BODY_NECK_KEYS = ['neck_0', 'angle_4', ]

TRAITS_TYPE_KEYS = ['type',]
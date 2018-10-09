import random
import numpy as np

material_colors_map = {'pink': {'200': '#F48FB1', '900': '#880E4F', '600': '#D81B60', 'A100': '#FF80AB', '300': '#F06292', 'A400': '#F50057', '700': '#C2185B', '50': '#FCE4EC', 'A700': '#C51162', '400': '#EC407A', '100': '#F8BBD0', '800': '#AD1457', 'A200': '#FF4081', '500': '#E91E63'}, 'blue': {'200': '#90CAF9', '900': '#0D47A1', '600': '#1E88E5', 'A100': '#82B1FF', '300': '#64B5F6', 'A400': '#2979FF', '700': '#1976D2', '50': '#E3F2FD', 'grey': '#263238', 'A700': '#2962FF', '400': '#42A5F5', '100': '#BBDEFB', '800': '#1565C0', 'A200': '#448AFF', '500': '#2196F3'}, 'indigo': {'200': '#9FA8DA', '900': '#1A237E', '600': '#3949AB', 'A100': '#8C9EFF', '300': '#7986CB', 'A400': '#3D5AFE', '700': '#303F9F', '50': '#E8EAF6', 'A700': '#304FFE', '400': '#5C6BC0', '100': '#C5CAE9', '800': '#283593', 'A200': '#536DFE', '500': '#3F51B5'}, 'brown': {'200': '#BCAAA4', '900': '#3E2723', '600': '#6D4C41', '300': '#A1887F', '700': '#5D4037', '50': '#EFEBE9', '400': '#8D6E63', '100': '#D7CCC8', '800': '#4E342E', '500': '#795548'}, 'purple': {'200': '#CE93D8', '900': '#4A148C', '600': '#8E24AA', 'A100': '#EA80FC', '300': '#BA68C8', 'A400': '#D500F9', '700': '#7B1FA2', '50': '#F3E5F5', 'A700': '#AA00FF', '400': '#AB47BC', '100': '#E1BEE7', '800': '#6A1B9A', 'A200': '#E040FB', '500': '#9C27B0'}, 'light': {'blue': '#0091EA', 'green': '#64DD17'}, 'grey': {'200': '#EEEEEE', '900': '#212121', '600': '#757575', '300': '#E0E0E0', '700': '#616161', '50': '#FAFAFA', '400': '#BDBDBD', '100': '#F5F5F5', '800': '#424242', '500': '#9E9E9E'}, 'deep': {'purple': '#6200EA', 'orange': '#DD2600'}, 'black': {'1000': '#000000'}, 'amber': {'200': '#FFE082', '900': '#FF6F00', '600': '#FFB300', 'A100': '#FFE57F', '300': '#FFD54F', 'A400': '#FFC400', '700': '#FFA000', '50': '#FFF8E1', 'A700': '#FFAB00', '400': '#FFCA28', '100': '#FFECB3', '800': '#FF8F00', 'A200': '#FFD740', '500': '#FFC107'}, 'green': {
    '200': '#A5D6A7', '900': '#1B5E20', '600': '#43A047', 'A100': '#B9F6CA', '300': '#81C784', 'A400': '#00E676', '700': '#388E3C', '50': '#E8F5E9', 'A700': '#00C853', '400': '#66BB6A', '100': '#C8E6C9', '800': '#2E7D32', 'A200': '#69F0AE', '500': '#4CAF50'}, 'yellow': {'200': '#FFF590', '900': '#F57F17', '600': '#FDD835', 'A100': '#FFFF82', '300': '#FFF176', 'A400': '#FFEA00', '700': '#FBC02D', '50': '#FFFDE7', 'A700': '#FFD600', '400': '#FFEE58', '100': '#FFF9C4', '800': '#F9A825', 'A200': '#FFFF00', '500': '#FFEB3B'}, 'teal': {'200': '#80CBC4', '900': '#004D40', '600': '#00897B', 'A100': '#A7FFEB', '300': '#4DB6AC', 'A400': '#1DE9B6', '700': '#00796B', '50': '#E0F2F1', 'A700': '#00BFA5', '400': '#26A69A', '100': '#B2DFDB', '800': '#00695C', 'A200': '#64FFDA', '500': '#009688'}, 'orange': {'200': '#FFCC80', '900': '#E65100', '600': '#FB8C00', 'A100': '#FFD180', '300': '#FFB74D', 'A400': '#FF9100', '700': '#F57C00', '50': '#FFF3E0', 'A700': '#FF6D00', '400': '#FFA726', '100': '#FFE0B2', '800': '#EF6C00', 'A200': '#FFAB40', '500': '#FF9800'}, 'cyan': {'200': '#80DEEA', '900': '#006064', '600': '#00ACC1', 'A100': '#84FFFF', '300': '#4DD0E1', 'A400': '#00E5FF', '700': '#0097A7', '50': '#E0F7FA', 'A700': '#00B8D4', '400': '#26C6DA', '100': '#B2EBF2', '800': '#00838F', 'A200': '#18FFFF', '500': '#00BCD4'}, 'white': {'500': '#ffffff'}, 'red': {'200': '#EF9A9A', '900': '#B71C1C', '600': '#E53935', 'A100': '#FF8A80', '300': '#E57373', 'A400': '#FF1744', '700': '#D32F2F', '50': '#FFEBEE', 'A700': '#D50000', '400': '#EF5350', '100': '#FFCDD2', '800': '#C62828', 'A200': '#FF5252', '500': '#F44336'}, 'lime': {'200': '#E6EE9C', '900': '#827717', '600': '#C0CA33', 'A100': '#F4FF81', '300': '#DCE775', 'A400': '#C6FF00', '700': '#A4B42B', '50': '#F9FBE7', 'A700': '#AEEA00', '400': '#D4E157', '100': '#F0F4C3', '800': '#9E9D24', 'A200': '#EEFF41', '500': '#CDDC39'}}


def getColor(name='', variant='500', out_type='BGRf', conversion_rate=1.0 / 255.0):

    if name == '':
        name = random.choice(material_colors_map.keys())

    color = material_colors_map[name][variant]

    if out_type == 'HEX':
        return color
    if out_type == 'RGB':
        color = color.lstrip("#")
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    if out_type == 'RGBf':
        color = color.lstrip("#")
        col = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        return np.array(col) * conversion_rate
    if out_type == 'BGR':
        color = color.lstrip("#")
        rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        return tuple(reversed(rgb))
    if out_type == 'BGRf':
        color = color.lstrip("#")
        rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        rgb = tuple(reversed(rgb))
        bgr = np.array(rgb) * conversion_rate
        return bgr


def getPalette(palette_name="default", conversion_rate=1.0 / 255.0):
    colors = []
    colors_names = ['red', 'green', 'blue', 'yellow',  'indigo', 'purple',  'teal', 'cyan', 'orange', 'white', 'lime', 'amber', 'pink', 'brown']
    for c in colors_names:
        colors.append(getColor(name=c))
    return colors


def getRandomColors():

    keys = material_colors_map.keys()
    random.shuffle(keys)
    print(keys)

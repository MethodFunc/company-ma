# Original ROI
# MK-SD33A ORG ROI(S11) - 201
# ROI = [(2, 4), (3, 4), (1, 5), (2, 5), (3, 5), (0, 6), (1, 6), (2, 6), (0, 7), (1, 7), (2, 7), (0, 8), (1, 8), (0, 9), (0, 10)]

#

# MK-SD33A NEW ROI(S43) - 201
# ROI = [(0, 5), (0, 7), (0, 8), (0, 9), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 5), (2, 6), (2, 10), (3, 3), (3, 5), (3, 9), (4, 7)]


# MK-SD33A fusionROI(S11+S43) - 201 fusion
# ROI = [(0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 5), (1, 6), (1, 7), (1, 8), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (4, 4)]

# MK-SD33C NEW ROI(S43) - 201
# ROI = [(8, 6), (9, 7), (6, 8), (7, 8), (9, 8), (6, 9), (7, 9), (6, 10),
#        (7, 10), (8, 10), (6, 11), (7, 11), (8, 11), (6, 12), (7, 12), (8, 12), (9, 12),
#        (6, 13), (7, 13), (8, 13), (9, 13), (6, 14), (7, 14), (8, 14), (9, 14),
#        (6, 15), (7, 15), (8, 15), (9, 15), (6, 16), (7, 16), (8, 16), (9, 16)]

# MK-SD33C fusionROI(S11+S43) - 201 fusion
# ROI = [(6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (9, 8), (9, 13), (9, 14), (9, 15)]

# MK-SD53R(weather_roi)(202)
# ROI = [(2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2)]

# MK-SD53R(road_roi)(201)
# ROI = [(0, 9), (0, 10), (0, 11), (0, 12), (1, 8), (1, 9), (1, 10), (1, 14), (1, 15), (2, 7), (2, 8), (2, 12), (2, 13),
#        (2, 14), (2, 15), (3, 6), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 4), (4, 8), (4, 9), (4, 10),
#        (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (5, 3), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (6, 5), (6, 6), (6, 7), (6, 8), (7, 3), (7, 4)]

def setting_roi(x):
    if x == '33A_201':
        ROI = [(0, 5), (0, 7), (0, 8), (0, 9), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 5), (2, 6), (2, 10), (3, 3),
               (3, 5), (3, 9), (4, 7)]

    if x == '33A_201_old':
        ROI = [(2, 4), (3, 4), (1, 5), (2, 5), (3, 5), (0, 6), (1, 6), (2, 6), (0, 7), (1, 7), (2, 7), (0, 8), (1, 8),
               (0, 9), (0, 10)]

    if x == '33A_201_fusion':
        ROI = [(0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 5), (1, 6), (1, 7), (1, 8), (2, 4), (2, 5), (2, 6), (2, 7),
               (3, 4), (3, 5), (4, 4)]

    if x == '33A_202':
        ROI = [(0, 8), (0, 9), (0, 10), (1, 8), (1, 9), (2, 1), (2, 7), (2, 8), (3, 0), (3, 1), (3, 6), (4, 0), (4, 1),
               (4, 5), (5, 0), (5, 1), (6, 0), (6, 1), (7, 1), (8, 1)]

    if x == '33C_201':
        ROI = [(8, 6), (9, 7), (6, 8), (7, 8), (9, 8), (6, 9), (7, 9), (6, 10), (7, 10), (8, 10), (6, 11), (7, 11),
               (8, 11), (6, 12), (7, 12), (8, 12), (9, 12), (6, 13), (7, 13), (8, 13), (9, 13), (6, 14), (7, 14),
               (8, 14), (9, 14), (6, 15), (7, 15), (8, 15), (9, 15), (6, 16), (7, 16), (8, 16), (9, 16)]

    if x == '33C_201_new':
        ROI = [(4, 7), (4, 8), (4, 9), (5, 7), (5, 8), (5, 9), (5, 10), (6, 7), (6, 8), (6, 9), (6, 10), (7, 5),
               (7, 10), (8, 6), (8, 7), (9, 7)]

    if x == '33C_201_fusion':
        ROI = [(6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (7, 9), (7, 10), (7, 11), (7, 12),
               (7, 13), (7, 14), (7, 15), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (9, 8), (9, 13), (9, 14),
               (9, 15)]

    if x == '53R_201':
        ROI = [(0, 9), (0, 10), (0, 11), (0, 12), (1, 8), (1, 9), (1, 10), (1, 14), (1, 15), (2, 7), (2, 8), (2, 12),
               (2, 13), (2, 14), (2, 15), (3, 6), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 4), (4, 8),
               (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (5, 3), (5, 7), (5, 8), (5, 9), (5, 10),
               (5, 11), (6, 5), (6, 6), (6, 7), (6, 8), (7, 3), (7, 4)]

    if x == '53R_202':
        ROI = [(2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
               (7, 1), (8, 1), (9, 1), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2)]

    return ROI


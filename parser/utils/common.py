# -*- coding: utf-8 -*-

pad = '<pad>'
unk = '<unk>'
bos = '<bos>'
eos = '<eos>'
nul = '<nul>'

pos_label = {"NN", "VV", "PU", "AD", "NR", "PN", "P", "CD", "M", "VA", "DEG", "JJ", "DEC", "VC", "NT", "SP", "DT", "LC",
             "CC", "AS", "VE", "IJ", "OD", "CS", "MSP", "BA", "DEV", "SB", "ETC", "DER", "LB", "IC", "NOI", "URL", "EM", "ON", "FW", "X"}

# coarse_productions = [
#     ('POS', 'POS*', 'POS*'),
#     ('POS*', 'POS*', 'POS*'),
#     ('SYN', 'POS', 'POS'),
#     ('SYN', 'POS', 'SYN'),
#     ('SYN', 'SYN', 'POS'),
#     ('SYN', 'SYN', 'SYN')
# ]

coarse_productions = [
    # ('POS', 'POS', 'SYN'),
    # ('POS', 'POS*', 'POS'),
    ('POS', 'POS*', 'POS*'),
    # ('POS', 'SYN', 'POS'),
    ('POS*', 'POS*', 'POS*'),
    # ('POS*', 'SYN', 'POS'),
    ('SYN', 'POS', 'POS'),
    ('SYN', 'POS', 'SYN'),
    ('SYN', 'SYN', 'POS'),
    ('SYN', 'SYN', 'SYN'),
    ('SYN', 'SYN*', 'POS'),
    ('SYN', 'SYN*', 'SYN'),
    # ('SYN', 'SYN*', 'SYN*'),
    ('SYN*', 'POS', 'POS'),
    ('SYN*', 'POS', 'SYN'),
    ('SYN*', 'SYN', 'POS'),
    ('SYN*', 'SYN', 'SYN'),
    ('SYN*', 'SYN*', 'POS'),
    ('SYN*', 'SYN*', 'SYN')
]

# ==================
#  from ctb51-big. modified by ctb7
# ==================

# coarse_productions = [
#     # ('POS', 'POS', 'SYN'),
#     ('POS', 'POS*', 'POS*'),
#     # ('POS', 'POS*', 'UnaryPOS'),
#     # ('POS', 'SYN', 'UnaryPOS'),
#     ('POS*', 'POS*', 'POS*'),
#     # ('POS*', 'SYN', 'UnaryPOS'),
#     ('SYN', 'POS', 'POS'),
#     ('SYN', 'POS', 'SYN'),
#     ('SYN', 'POS', 'UnaryPOS'),
#     ('SYN', 'POS', 'UnarySYN'),
#     ('SYN', 'SYN', 'POS'),
#     ('SYN', 'SYN', 'SYN'),
#     ('SYN', 'SYN', 'UnaryPOS'),
#     ('SYN', 'SYN', 'UnarySYN'),
#     ('SYN', 'SYN*', 'POS'),
#     ('SYN', 'SYN*', 'SYN'),
#     ('SYN', 'SYN*', 'UnaryPOS'),
#     ('SYN', 'SYN*', 'UnarySYN'),
#     ('SYN', 'UnaryPOS', 'POS'),
#     ('SYN', 'UnaryPOS', 'SYN'),
#     ('SYN', 'UnaryPOS', 'UnaryPOS'),
#     ('SYN', 'UnaryPOS', 'UnarySYN'),
#     ('SYN', 'UnarySYN', 'POS'),
#     ('SYN', 'UnarySYN', 'SYN'),
#     ('SYN', 'UnarySYN', 'UnaryPOS'),
#     ('SYN', 'UnarySYN', 'UnarySYN'),
#     ('SYN*', 'POS', 'POS'),
#     ('SYN*', 'POS', 'SYN'),
#     ('SYN*', 'POS', 'UnaryPOS'),
#     ('SYN*', 'POS', 'UnarySYN'),
#     ('SYN*', 'SYN', 'POS'),
#     ('SYN*', 'SYN', 'SYN'),
#     ('SYN*', 'SYN', 'UnaryPOS'),
#     ('SYN*', 'SYN', 'UnarySYN'),
#     ('SYN*', 'SYN*', 'POS'),
#     ('SYN*', 'SYN*', 'SYN'),
#     ('SYN*', 'SYN*', 'UnaryPOS'),
#     ('SYN*', 'SYN*', 'UnarySYN'),
#     ('SYN*', 'UnaryPOS', 'POS'),
#     ('SYN*', 'UnaryPOS', 'SYN'),
#     ('SYN*', 'UnaryPOS', 'UnaryPOS'),
#     ('SYN*', 'UnaryPOS', 'UnarySYN'),
#     ('SYN*', 'UnarySYN', 'POS'),
#     ('SYN*', 'UnarySYN', 'SYN'),
#     ('SYN*', 'UnarySYN', 'UnaryPOS'),
#     ('SYN*', 'UnarySYN', 'UnarySYN'),
#     ('UnaryPOS', 'POS*', 'POS*'),
#     ('UnarySYN', 'POS', 'POS'),
#     ('UnarySYN', 'POS', 'SYN'),
#     ('UnarySYN', 'POS', 'UnaryPOS'),
#     ('UnarySYN', 'POS', 'UnarySYN'),
#     ('UnarySYN', 'SYN', 'POS'),
#     ('UnarySYN', 'SYN', 'SYN'),
#     ('UnarySYN', 'SYN', 'UnaryPOS'),
#     ('UnarySYN', 'SYN', 'UnarySYN'),
#     ('UnarySYN', 'SYN*', 'POS'),
#     ('UnarySYN', 'SYN*', 'SYN'),
#     # ('UnarySYN', 'SYN*', 'SYN*'),
#     ('UnarySYN', 'SYN*', 'UnaryPOS'),
#     ('UnarySYN', 'SYN*', 'UnarySYN'),
#     ('UnarySYN', 'UnaryPOS', 'POS'),
#     ('UnarySYN', 'UnaryPOS', 'SYN'),
#     ('UnarySYN', 'UnaryPOS', 'UnaryPOS'),
#     ('UnarySYN', 'UnaryPOS', 'UnarySYN'),
#     ('UnarySYN', 'UnarySYN', 'POS'),
#     ('UnarySYN', 'UnarySYN', 'SYN'),
#     ('UnarySYN', 'UnarySYN', 'UnaryPOS'),
#     ('UnarySYN', 'UnarySYN', 'UnarySYN')
# ]


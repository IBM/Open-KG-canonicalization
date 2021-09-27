#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ====== Random Utility Functions ======
def invertDict(curr_map):
    if not curr_map: return dict()
    entry = list(curr_map.items())[0]
    if type(entry[0]) in [str, int] and type(entry[1]) in [list, set]:
        inv_map = dict()
        for key, value_collection in curr_map.items():
            for value in value_collection:
                inv_map[value] = inv_map.get(value, set())
                inv_map[value].add(key)
    elif all(map(lambda z: type(z) is str, entry)):
        inv_map = dict()
        for key, value in curr_map.items():
            inv_map[value] = inv_map.get(value, set())
            inv_map[value].add(key)
    else:
        raise NotImplementedError('{0}:{1}'.format(type(entry[0]), type(entry[1])))
    return inv_map

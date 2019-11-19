#! /usr/bin/python
# -*- coding: utf-8 -*-

from satpy import Scene
import .textures

def textures_for_scene(scn, textures, append=True):
    """
    Wrapper to calculate multiple textures for datasets in a
    :class:`satpy.scene.Scene`.

    Parameters
    ----------
    scn : satpy.Scene
    textures : dict
        Dictionary with textures bands to calulate. The accepted notation is
        {("texture name", lag, win_size, win_geom): [list of dataset names or tuples of dataset names]}

        For example:
            {("variogram", 2, 7, "round"), ["IR_039", "IR_103"]}
    append : boolean, optional
        If `False` returns a new :class:`satpy.scene.Scene` with all calculated textures,
        By default returns a new :class:`satpy.scene.Scene` with all input datasets and
        all calculated textures.

    Returns
    -------
    satpy.Scene

    """
    if append:
        out_scn = scn.copy()
    else:
        out_scn = Scene()

    for tex, bands in textures.items():
        tex_name, lag, win_size, win_geom = tex
        fun = getattr(textures, tex_name)

        for b in bands:
            if tex_name in ["cross_variogram", "pseudo_cross_variogram"]:
                x, y = b
                tex_res = fun(scn[x], scn[y], lag, win_size, win_geom)
            else:
                tex_res = fun(scn[b], lag, win_size, win_geom)

            out_scn[tex_res.name] = tex_res

    return out_scn

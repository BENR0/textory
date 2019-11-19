#! /usr/bin/python
# -*- coding: utf-8 -*-

import textory.textures as txt
from satpy import Scene

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
    #attributes to strip to satpy scene datasets
    strip_attrs = ["calibration", "standard_name", "units", "name", "wavelength"]

    if append:
        out_scn = scn.copy()
    else:
        out_scn = Scene()

    for tex, bands in textures.items():
        tex_name, lag, win_size, win_geom = tex
        fun = getattr(txt, tex_name)

        for b in bands:
            if tex_name in ["cross_variogram", "pseudo_cross_variogram"]:
                x, y = b
                tex_res = fun(scn[x], scn[y], lag=lag, win_size=win_size, win_geom=win_geom)
            else:
                tex_res = fun(scn[b], lag=lag, win_size=win_size, win_geom=win_geom)

            for k in strip_attrs:
                tex_res.attrs.pop(k)

            out_scn[tex_res.name] = tex_res

    return out_scn


def textures_for_xr_dataset(xrds, textures, append=True):
    """
    Wrapper to calculate multiple textures for dataarrays in a
    :class:`xarray.Dataset`.

    Parameters
    ----------
    xrds : :class:`xarray.Dataset`
    textures : dict
        Dictionary with textures bands to calulate. The accepted notation is
        {("texture name", lag, win_size, win_geom): [list of dataset names or tuples of dataset names]}

        For example:
            {("variogram", 2, 7, "round"), ["IR_039", "IR_103"]}
    append : boolean, optional
        If `False` returns a new :class:`xarray.Dataset` with all calculated textures,
        By default returns a new :class:`xarray.Dataset` with all input datasets and
        all calculated textures.

    Returns
    -------
    satpy.Scene

    """
    out_ds = xrds.copy()
    if not append:
        var_names = [name for name, _ in ds.data_vars.items()]
        out_ds = out_ds.drop(var_names)

    for tex, bands in textures.items():
        tex_name, lag, win_size, win_geom = tex
        fun = getattr(txt, tex_name)

        for b in bands:
            if tex_name in ["cross_variogram", "pseudo_cross_variogram"]:
                x, y = b
                tex_res = fun(xrds[x], xrds[y], lag=lag, win_size=win_size, win_geom=win_geom)
            else:
                tex_res = fun(xrds[b], lag=lag, win_size=win_size, win_geom=win_geom)

            out_ds[tex_res.name] = tex_res

    return out_ds

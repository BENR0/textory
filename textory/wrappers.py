#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Textory satpy Scene and xarray Dataset wrappers

Calculate textures for Scene
----------------------------

With the :func:`~textory.wrappers.textures_for_scene` function it is easy to
calculate one or multiple textures with the same or different parameters for
datasets in a :class:`satpy.scene.Scene`. The function will return a new Scene
either with the textures in addition to all input datasets (default) or a Scene only with
the textures, depending on the ``append`` parameter of the function.

The ``textures`` parameter takes a dictionary where the keys are a tuple with the texture
and the parameters which to calculate and the values are a list (or list of tuples in the
case of textures which require two inputs) of the datasets to apply the texture to in the general
form of:

.. code-block:: python

    textures_dict = {("texture_name", lag, win_size, win_geom): [list of dataset names]}

The following example would calculate the variogram with lag=2, win_size=7, win_geom="square"
for the datasets with name "IR_039" and "IR_108" as well as the cross variogram with lag=1,
win_size=5, and win_geom="round" between the datasets with name "WV_062" and "IR_108" of the Scene.

.. code-block:: python

    import textory as tx

    scn = Scene(...)
    textures_dict = {("variogram", 2, 7, "square"): ["IR_039", "IR_108"],
                     ("cross_variogram", 1, 5, "round"): [("WV_062", "IR_108")]}
    scn_with_textures = tx.textures_for_scene(scn, textures=textures_dict)

Calculate textures for xarray Dataset
-------------------------------------

The :func:`~textory.wrappers.textures_for_xr_dataset` function works similarly to the
:func:`~textory.wrappers.textures_for_scene` function above but takes :class:`xarray.Dataset`
as input and also returns a :class:`xarray.Dataset`.
"""
import textory.textures as txt


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
        for b in bands:
            if tex[0] == "window_statistic":
                tex_name, stat, win_size = tex
                fun = getattr(txt, tex_name)
                tex_res = fun(scn[b], stat=stat, win_size=win_size)
            else:
                tex_name, lag, win_size, win_geom = tex
                fun = getattr(txt, tex_name)

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
    :class:`xarray.Dataset`

    """
    out_ds = xrds.copy()
    if not append:
        var_names = [name for name, _ in out_ds.data_vars.items()]
        out_ds = out_ds.drop(var_names)

    for tex, bands in textures.items():
        for b in bands:
            if tex[0] == "window_statistic":
                tex_name, stat, win_size = tex
                fun = getattr(txt, tex_name)
                tex_res = fun(out_ds[b], stat=stat, win_size=win_size)
            else:
                tex_name, lag, win_size, win_geom = tex
                fun = getattr(txt, tex_name)

                if tex_name in ["cross_variogram", "pseudo_cross_variogram"]:
                    x, y = b
                    tex_res = fun(out_ds[x], out_ds[y], lag=lag, win_size=win_size, win_geom=win_geom)
                else:
                    tex_res = fun(out_ds[b], lag=lag, win_size=win_size, win_geom=win_geom)

            out_ds[tex_res.name] = tex_res

    return out_ds

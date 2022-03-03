from matplotlib import pyplot as plt
import numpy as np
from gammapy.maps import MapAxis
import matplotlib.colors as colors

def plot_alpha_map(bg, path="./"):
    plt.figure()
    plt.pcolormesh(
        bg.lon_axis.edges.value,
        bg.lat_axis.edges.value,
        bg.alpha_map,
        vmin=0,
        vmax=1,
        rasterized=True
    )
    plt.colorbar(label="Alpha = exposure_effective / exposure_observed")
    plt.xlabel("Lon / deg ")
    plt.ylabel("Lat / deg ")
    plt.savefig(path + "alpha_map.pdf")


def plot_counts_maps(bg, path="./"):
    counts_eff = np.sum(bg.counts_map_eff, axis=0)
    counts_obs = np.sum(bg.counts_map_obs, axis=0)
    counts_corr = np.sum(bg.counts_map_eff, axis=0) / bg.alpha_map

    counts_max = np.max([
        counts_eff[np.isfinite(counts_eff)].max(),
        counts_obs[np.isfinite(counts_obs)].max(),
        counts_corr[np.isfinite(counts_corr)].max().value
    ])

    plt.figure()
    im = plt.pcolormesh(
        bg.lon_axis.edges.value,
        bg.lat_axis.edges.value,
        counts_eff,
        vmin = 0,
        vmax = counts_max
    )
    im.set_rasterized(True)
    plt.colorbar(label="Effective Counts")
    plt.xlabel("Lon / deg ")
    plt.ylabel("Lat / deg ")
    plt.savefig(path + "counts_map_eff.pdf")

    plt.figure()
    im = plt.pcolormesh(
        bg.lon_axis.edges.value,
        bg.lat_axis.edges.value,
        counts_obs,
        vmin = 0,
        vmax = counts_max
    )
    im.set_rasterized(True)
    plt.colorbar(label="Observed Counts")
    plt.xlabel("Lon / deg ")
    plt.ylabel("Lat / deg ")
    plt.savefig(path + "counts_map_obs.pdf")

    plt.figure()
    im = plt.pcolormesh(
        bg.lon_axis.edges.value,
        bg.lat_axis.edges.value,
        counts_corr,
        vmin = 0,
        vmax = counts_max
    )
    im.set_rasterized(True)
    plt.colorbar(label="Corrected Counts")
    plt.xlabel("Lon / deg ")
    plt.ylabel("Lat / deg ")
    plt.savefig(path + "counts_map_corr.pdf")




def plot_Background2D(bg, path="./"):
    nbins=200
    rmin = bg.offset.edges[:-1]
    rmax = bg.offset.edges[1:]
    # calculate local radius map for this binning
    lon_axis = MapAxis.from_bounds(
        -bg.offset_max.value,
        bg.offset_max.value,
        nbins,
        interp="lin",
        unit="deg",
        name="fov_lon",
    )
    lat_axis = MapAxis.from_bounds(
        -bg.offset_max.value,
        bg.offset_max.value,
        nbins,
        interp="lin",
        unit="deg",
        name="fov_lat",
    )
    lon, lat = np.meshgrid(lon_axis.center.value, lat_axis.center.value)
    radius_map = np.sqrt(lon ** 2 + lat ** 2)
    #plotting
    fig, axs = plt.subplots(bg.e_reco.nbin, 1, figsize=(5, 4*bg.e_reco.nbin), constrained_layout=True)
    for i, bg_rate in enumerate(bg.bg_rate):
        z = np.zeros((200,200))
        z[:] = np.nan
        j = 0
        for rmi, rma in zip(rmin.value, rmax.value):
            mask = (radius_map >= rmi) & (radius_map < rma)
            z[mask] = bg_rate[j]
            j += 1
        axs[i].set_title(
            f"{bg.e_reco.edges[i]:.2f} $<E<$ {bg.e_reco.edges[i+1]:.2f}"
        )
        im = axs[i].pcolormesh(
            lon_axis.edges.value,
            lat_axis.edges.value,
            z,
            norm=colors.LogNorm(
                vmin=np.min([b.value for b in bg.bg_rate]),
                vmax=np.max([b.value for b in bg.bg_rate]),
            ),
            rasterized=True
        )
        axs[i].set(aspect="equal")
        axs[i].set_xlim(
            bg.lon_axis.edges.min().value, bg.lon_axis.edges.max().value
        )
        axs[i].set_ylim(
            bg.lon_axis.edges.min().value, bg.lon_axis.edges.max().value
        )
        axs[i].set_xlabel("Lon / deg")
        axs[i].set_ylabel("Lat / deg")
    fig.colorbar(im, ax=axs[:bg.e_reco.nbin], label="Background rate / (s$-1$ MeV-1 sr-1)", location="right")
    plt.savefig(path + "background_overview_2D.pdf")


def plot_Background3D(bg, path="./"):
    fig, axs = plt.subplots(bg.e_reco.nbin, 1, figsize=(5, 4*bg.e_reco.nbin), constrained_layout=True)
    for i, bg_rate in enumerate(bg.get_bg_3d().data):
        axs[i].set_title(
            f"{bg.e_reco.edges[i]:.2f} $<E<$ {bg.e_reco.edges[i+1]:.2f}"
            )
        im = axs[i].pcolormesh(
            bg.lon_axis.edges.value,
            bg.lat_axis.edges.value,
            bg_rate.value,
            norm=colors.LogNorm(
                vmin=np.min([b.value for b in bg.bg_rate]),
                vmax=np.max([b.value for b in bg.bg_rate]),
            ),
            rasterized=True
        )
        axs[i].set(aspect="equal")
        axs[i].set_xlabel("Lon / deg")
        axs[i].set_ylabel("Lat / deg")
    fig.colorbar(im, ax=axs[:bg.e_reco.nbin], label="Background rate / (s$-1$ MeV-1 sr-1)", location="right")
    plt.savefig(path + "background_overview_3D.pdf")

def plot_Background2D3D(bg, path="./", nbins=200):
    #### 2D ###
    rmin = bg.offset.edges[:-1]
    rmax = bg.offset.edges[1:]
    # calculate local radius map for this binning
    lon_axis = MapAxis.from_bounds(
        -bg.offset_max.value,
        bg.offset_max.value,
        nbins,
        interp="lin",
        unit="deg",
        name="fov_lon",
    )
    lat_axis = MapAxis.from_bounds(
        -bg.offset_max.value,
        bg.offset_max.value,
        nbins,
        interp="lin",
        unit="deg",
        name="fov_lat",
    )
    lon, lat = np.meshgrid(lon_axis.center.value, lat_axis.center.value)
    radius_map = np.sqrt(lon ** 2 + lat ** 2)
    #plotting
    fig, axs = plt.subplots(2, bg.e_reco.nbin, figsize=(4*bg.e_reco.nbin, 8), constrained_layout=True)
    for i, bg_rate in enumerate(bg.bg_rate):
        z = np.zeros((200,200))
        z[:] = np.nan
        j = 0
        for rmi, rma in zip(rmin.value, rmax.value):
            mask = (radius_map >= rmi) & (radius_map < rma)
            z[mask] = bg_rate[j]
            j += 1
        axs[0,i].set_title(
            f"{bg.e_reco.edges[i]:.2f} $<E<$ {bg.e_reco.edges[i+1]:.2f}",
            size=13
        )
        im = axs[0,i].pcolormesh(
            lon_axis.edges.value,
            lat_axis.edges.value,
            z,
            norm=colors.LogNorm(
                vmin=np.min([b.value for b in bg.bg_rate]),
                vmax=np.max([b.value for b in bg.bg_rate]),
            ),
            rasterized=True
        )
        axs[0,i].set(aspect="equal")
        axs[0,i].set_ylabel("Lat / deg")
    fig.colorbar(im, ax=axs[:2,:], label="Background rate / (s$-1$ MeV-1 sr-1)", location="right")
    ### 3D ###
    for i, bg_rate in enumerate(bg.get_bg_3d().data):
        im = axs[1,i].pcolormesh(
            bg.lon_axis.edges.value,
            bg.lat_axis.edges.value,
            bg_rate.value,
            norm=colors.LogNorm(
                vmin=np.min([b.value for b in bg.bg_rate]),
                vmax=np.max([b.value for b in bg.bg_rate]),
            ),
            rasterized=True
        )
        axs[1,i].set(aspect="equal")
        axs[1,i].set_xlabel("Lon / deg")
        axs[1,i].set_ylabel("Lat / deg")
    plt.savefig(path + "background_overview_2D3D.pdf")
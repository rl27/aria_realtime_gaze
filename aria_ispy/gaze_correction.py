def apply_affine_correction(tv_xy, correction):
    if tv_xy is None or correction is None:
        return tv_xy

    x, y = tv_xy
    ax, ay = correction
    return [
        float(ax[0] * x + ax[1] * y + ax[2]),
        float(ay[0] * x + ay[1] * y + ay[2]),
    ]

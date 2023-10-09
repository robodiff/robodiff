def ramp(x, slope=1/12, xoffset=0.0, yoffset=-2.5/128):
    return (x-xoffset) * slope + yoffset


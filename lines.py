import numpy as np
from perlin import generate_2D_perlin_noise

def normalize(x):
    return x / np.sqrt(np.dot(x, x))

def perpendicular(x):
    normed = normalize(x)
    return np.array([normed[1], -normed[0]])

def generate_bates(n, a, b, d):
    if d > 0:
        return np.array([np.mean(np.random.uniform(a, b, abs(d))) for i in range(n)])
    gens = []
    mean = (b - a) / 2
    for i in range(n):
        result = np.mean(np.random.uniform(a, b, abs(d)))
        if result < mean:
            result += mean
        else:
            result -= mean
        gens.append(result)
    return np.array(gens)

def generate_widths(n, limits, pattern):
    widths = [np.random.uniform(limits[0], limits[1])]
    counter = pattern - 1
    for i in range(n-2):
        if counter > 0:
            widths.append(widths[-1])
        else:
            widths.append(np.random.uniform(limits[0], limits[1]))
            counter = pattern
        counter -= 1
    return widths

def shuffle_points(ps, pattern):
    i = 0
    j = 0
    new_ps = [ps[0]]
    used_last = False
    while i + pattern[j % len(pattern)] < len(ps):
        i += pattern[j % len(pattern)]
        new_ps.append(ps[i])
        used_last = used_last or i == (len(ps) - 1)
        j += 1
    if not used_last:
        new_ps.append(ps[-1])
    return np.array(new_ps)

def divide_points(p1, p2, n, d):
    shifts = np.sort(generate_bates(n-2, 0, 1, d))
    xs = np.array([p1[0] + shift * (p2[0] - p1[0]) for shift in shifts]).reshape(n-2, 1)
    ys = np.array([p1[1] + shift * (p2[1] - p1[1]) for shift in shifts]).reshape(n-2, 1)
    mid_ps = np.concatenate((xs, ys), axis=1)
    return np.concatenate((np.array(p1).reshape(1,2), mid_ps, np.array(p2).reshape(1,2)), axis=0)

def move_points(ps, scale):
    perpend = perpendicular(ps[-1] - ps[0])
    new_ps = np.copy(ps)
    for i in range(1, len(new_ps) - 1):
        new_ps[i] += np.random.uniform(-scale, scale) * perpend
    return new_ps

def double_points(ps, scales):
    left = np.copy(ps)
    right = np.copy(ps)
    for i in range(len(ps) - 1):
        shift = perpendicular(ps[i+1]-ps[i])
        left[i] += shift * scales[i]
        right[i] -= shift * scales[i]
    shift = perpendicular(ps[-1]-ps[-2])
    left[-1] += shift * scales[-1]
    right[-1] -= shift * scales[-1]
    return left, right

def catmull_rom_matrix(t):
    return np.array([
        [0, 1, 0, 0],
        [-t, 0, t, 0],
        [2*t, t - 3, 3 - 2 * t, -t],
        [-t, 2 - t, t - 2, t]
    ])

cr_matrix = catmull_rom_matrix(0.5)

def spline(ps):
    xs = np.dot(cr_matrix, ps[:,0])
    ys = np.dot(cr_matrix, ps[:,1])
    def compute_spline(t):
        ts = [1, t, t * t, t ** 3]
        return [np.dot(ts, xs), np.dot(ts, ys)]

    return compute_spline

def draw_3_spline(cr, ps, scales, d, left):
    new_ps = np.concatenate((ps[:1], ps)) if left else np.concatenate((ps, ps[-1:]))
    draw_spline(cr, new_ps, scales, d)

def draw_spline(cr, ps, scales, d):
    ts = np.linspace(0,1,int(5/0.1))
    ws = np.concatenate((np.linspace(scales[0], scales[1], 20), scales[1] * np.ones(30)))
    s = spline(ps)
    
    ps = np.array([s(t) for t in ts])
    left, right = double_points(ps, ws)
    
    
    cr.set_line_width(0.001)
    cr.move_to(left[0,0], left[0,1])
    for i in range(1, len(left)):
        cr.line_to(left[i,0], left[i,1])
    
    for i in reversed(range(len(right))):
        cr.line_to(right[i,0], right[i,1])
    cr.line_to(left[0,0], left[0,1])
    cr.fill()
    cr.stroke()
        
def scaled_perlin(n, a, b):
    perlin = generate_2D_perlin_noise(n, n)
    mn = np.min(perlin)
    mx = np.max(perlin)
    scaled = (perlin - mn) / (mx - mn) * (b - a) + a
    return scaled

def scale_grid(p):
    return [0.05 + p[0] * 0.9, 0.05 + p[1] * 0.9]

def sketchy_line(cr, p1, p2, n, ms, ws, wsp, sp, d):
    """
    cr - pyton drawing lib context
    
    p1 - from point
    p2 - to point
    
    n - the number of control points
    ms - the amount of displacement on each control point
    ws -  the stroke width to be used
    wsp - the count of points for which a stroke retains its width
    sp - the order in which the control points are connected
    d -  the distribution of the control points along the line
    """
    ps = shuffle_points(move_points(divide_points(p1, p2, n, d), ms), sp)
    widths = generate_widths(len(ps), ws, wsp)

    draw_3_spline(cr, ps[:3], [widths[0], widths[0]], 0.1, True)
    for i in range(len(ps)-3):
        draw_spline(cr, ps[i:i+4], [widths[i], widths[i+1]], 0.1)
    draw_3_spline(cr, ps[-3:], [widths[-2], widths[-1]], 0.1, False)

def sketchy_grid(cr, n, p, m, w, d):
    """
    cr - pyton drawing lib context
    
    n - grid size
    
    p - limits for number of control points
    m - limits for displacements
    w -  limits for width
    d -  limits for bates distribution parameter 
    """
    perlin_p = scaled_perlin(n+1, p[0], p[1])
    perlin_m = scaled_perlin(n+1, m[0], m[1])
    perlin_w = scaled_perlin(n+1, w[0], w[1])
    perlin_d = scaled_perlin(n+1, d[0], d[1])
    for i in range(n+1):
        for j in range(n):
            p0 = scale_grid([j/(n+1), i/(n+1)])
            p1 = scale_grid([(j+1)/(n+1), i/(n+1)])
            
            pd_val = round(perlin_d[i, j])
            pd_val = pd_val if pd_val != 0 else 1
            
            sketchy_line(cr, p0, p1, round(perlin_p[i,j]), perlin_m[i,j], [perlin_w[i,j], perlin_w[i,j]], 
                         1, [1], pd_val)
    for i in range(n):
        for j in range(n+1):
            p0 = scale_grid([j/(n+1), i/(n+1)])
            p1 = scale_grid([j/(n+1), (i+1)/(n+1)])
            
            pd_val = round(perlin_d[i, j])
            pd_val = pd_val if pd_val != 0 else 1
            
            sketchy_line(cr, p0, p1, round(perlin_p[i,j]), perlin_m[i,j], [perlin_w[i,j], perlin_w[i,j]], 
                         1, [1], pd_val)
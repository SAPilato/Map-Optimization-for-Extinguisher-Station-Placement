from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pulp
from sklearn.cluster import KMeans
import warnings
import time
warnings.filterwarnings("ignore")

IMG_PATH       = 'Greenwell-map.png' # change to fit your file name
CELL           = 40      # resolution, increase this if stalling (20=slow, 40=fast, 60=very fast)
THRESHOLD      = 80      # keep this at roughly CELL² × 0.05
S              = 75      # coverage radius in pixels
P              = 40      # number of stations

t_total = time.time()

print("─" * 60)
print(f"[1/5] Loading image: {IMG_PATH}")
t = time.time()
img = Image.open(IMG_PATH).convert('RGB')
arr = np.array(img)
H, W = arr.shape[:2]
r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
yellow = (r > 200) & (g > 160) & (b < 80) & (r > b + 140) & (g > b + 90)
print(f"    Image size: {W}x{H}px  |  Yellow pixels found: {yellow.sum():,}  ({time.time()-t:.1f}s)")

print(f"\n[2/5] Building demand points  (cell={CELL}px, threshold={THRESHOLD})")
t = time.time()
gh, gw = H // CELL, W // CELL
demand_points = []
for gy in range(gh):
    for gx in range(gw):
        weight = int(yellow[gy*CELL:(gy+1)*CELL, gx*CELL:(gx+1)*CELL].sum())
        if weight > THRESHOLD:
            cx = gx * CELL + CELL // 2
            cy = gy * CELL + CELL // 2
            demand_points.append((cx, cy, weight))

n_demand = len(demand_points)
candidates = [(x, y) for x, y, w in demand_points]
weights    = [w for _, _, w in demand_points]
coords     = np.array(candidates)
n_cands    = len(candidates)
print(f"    Demand points: {n_demand}  |  Candidates: {n_cands}  ({time.time()-t:.1f}s)")

if n_demand == 0:
    print("\n  ERROR: No demand points found.")
    print("  Fix: lower THRESHOLD or increase CELL size.")
    exit()

if n_demand > 400:
    print(f"\n  WARNING: {n_demand} demand points.")
    print(f"  ILP algorithms may be slow. Consider CELL=60, THRESHOLD=180.")


print(f"\n[3/5] Building coverage matrix  ({n_demand}x{n_cands} pairs)...")
t = time.time()
coverage = {}
for i, (xi, yi, _) in enumerate(demand_points):
    coverage[i] = []
    for j, (xj, yj) in enumerate(candidates):
        if (xi-xj)**2 + (yi-yj)**2 <= S*S:
            coverage[i].append(j)
    if i % 100 == 0 and i > 0:
        elapsed = time.time() - t
        eta = elapsed / i * (n_demand - i)
        print(f"    {i}/{n_demand} ({100*i/n_demand:.0f}%)  elapsed: {elapsed:.0f}s  eta: {eta:.0f}s")

print(f"    Coverage matrix done  ({time.time()-t:.1f}s)")

print("    Building distance matrix...")
t = time.time()
CAP = S * 4
dist_matrix = np.zeros((n_demand, n_cands), dtype=np.float32)
for i, (xi, yi, _) in enumerate(demand_points):
    for j, (xj, yj) in enumerate(candidates):
        dist_matrix[i, j] = min(np.sqrt((xi-xj)**2 + (yi-yj)**2), CAP)
print(f"    Distance matrix done  ({time.time()-t:.1f}s)")

# Drawing for stations

def draw_result(chosen_indices, title, filename, color=(210, 30, 30)):
    chosen_coords = [candidates[j] for j in chosen_indices]
    img_out = Image.open(IMG_PATH).convert('RGBA')
    overlay = Image.new('RGBA', img_out.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    cr, cg, cb = color
    for cx, cy in chosen_coords:
        od.ellipse([cx-S, cy-S, cx+S, cy+S],
                   fill=(cr, cg, cb, 40), outline=(cr//2, cg//2, cb//2, 150), width=2)
    result = Image.alpha_composite(img_out, overlay).convert('RGB')
    draw   = ImageDraw.Draw(result)
    try:
        font_title  = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        font_label  = ImageFont.truetype("DejaVuSans-Bold.ttf", 11)
        font_legend = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font_title = font_label = font_legend = ImageFont.load_default()
    for i, (cx, cy) in enumerate(chosen_coords):
        draw.ellipse([cx-15, cy-15, cx+15, cy+15], fill=(255, 255, 255))
        draw.ellipse([cx-12, cy-12, cx+12, cy+12],
                     fill=color, outline=(cr//2, cg//2, cb//2), width=2)
        lbl = f"S{i+1}"
        bb  = font_label.getbbox(lbl)
        draw.text((cx-(bb[2]-bb[0])//2, cy-(bb[3]-bb[1])//2-1),
                  lbl, fill=(255, 255, 255), font=font_label)
    covered_count = sum(
        1 for i, (xi, yi, _) in enumerate(demand_points)
        if any((xi-cx)**2+(yi-cy)**2 <= S*S for cx, cy in chosen_coords)
    )
    pct = 100 * covered_count / n_demand
    draw.rectangle([0, 0, W, 52], fill=(30, 30, 30))
    draw.text((16, 10), title, fill=(255, 255, 255), font=font_title)
    draw.text((W-340, 16),
              f"p={len(chosen_coords)}  |  {covered_count}/{n_demand} zones  |  {pct:.1f}% coverage",
              fill=(200, 200, 200), font=font_legend)
    result.save(filename)
    print(f"    Saved -> {filename}  ({covered_count}/{n_demand} zones, {pct:.1f}%)")
    return covered_count, pct

print(f"\n[4/5] Running algorithms  (P={P} stations)\n")

# ILP

print("── Algorithm 1: ILP ──")
t = time.time()
print(f"    Building model ({n_cands + n_demand} variables)...")
model = pulp.LpProblem("ILP", pulp.LpMaximize)
x = [pulp.LpVariable(f"x_{j}", cat='Binary') for j in range(n_cands)]
y = [pulp.LpVariable(f"y_{i}", cat='Binary') for i in range(n_demand)]
model += pulp.lpSum(weights[i] * y[i] for i in range(n_demand))
model += pulp.lpSum(x) <= P
for i in range(n_demand):
    if coverage[i]:
        model += y[i] <= pulp.lpSum(x[j] for j in coverage[i])
    else:
        model += y[i] == 0
    if i % 200 == 0 and i > 0:
        print(f"    Constraints: {i}/{n_demand}  ({time.time()-t:.0f}s elapsed)")
print(f"    Model built ({time.time()-t:.1f}s) — starting solver...")
model.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=180))
print(f"    Solver done ({time.time()-t:.1f}s total)")
ilp_chosen = [j for j in range(n_cands) if x[j].value() and x[j].value() > 0.5]
draw_result(ilp_chosen, "Algorithm 1 - ILP", "algo1_ilp.png", color=(210, 30, 30))


#  K-MEANS

print("\n── Algorithm 2: K-Means Clustering ──")
t = time.time()
print(f"    Fitting k-means (k={P})...")
km = KMeans(n_clusters=P, n_init=20, random_state=42)
km.fit(coords, sample_weight=weights)
centres = km.cluster_centers_
kmeans_chosen = []
for cx, cy in centres:
    dists   = np.sqrt((coords[:,0]-cx)**2 + (coords[:,1]-cy)**2)
    nearest = int(np.argmin(dists))
    if nearest not in kmeans_chosen:
        kmeans_chosen.append(nearest)
print(f"    Done ({time.time()-t:.1f}s)")
draw_result(kmeans_chosen, "Algorithm 2 - K-Means Clustering", "algo2_kmeans.png", color=(30, 170, 80))


# P-MEDIAN

print("\n── Algorithm 3: P-Median (fast approximation) ──")
t = time.time()
print("    Initialising with k-means centres...")
km_init = KMeans(n_clusters=P, n_init=10, random_state=42)
km_init.fit(coords, sample_weight=weights)
current_centres = km_init.cluster_centers_.copy()

for iteration in range(50):
    dists_all   = np.sqrt(((coords[:,None,:] - current_centres[None,:,:])**2).sum(axis=2))
    assignments = np.argmin(dists_all, axis=1)
    new_centres = np.zeros_like(current_centres)
    for k in range(P):
        members = np.where(assignments == k)[0]
        if len(members) == 0:
            new_centres[k] = current_centres[k]
            continue
        w = np.array([weights[i] for i in members])
        new_centres[k] = np.average(coords[members], axis=0, weights=w)
    shift = np.sqrt(((new_centres - current_centres)**2).sum(axis=1)).max()
    current_centres = new_centres
    if iteration % 10 == 0:
        print(f"    Iteration {iteration+1}  —  max shift: {shift:.2f}px  ({time.time()-t:.0f}s)")
    if shift < 0.5:
        print(f"    Converged after {iteration+1} iterations")
        break

pmedian_chosen = []
for cx, cy in current_centres:
    dists   = np.sqrt((coords[:,0]-cx)**2 + (coords[:,1]-cy)**2)
    nearest = int(np.argmin(dists))
    if nearest not in pmedian_chosen:
        pmedian_chosen.append(nearest)
print(f"    Done ({time.time()-t:.1f}s)")
draw_result(pmedian_chosen, "Algorithm 3 - P-Median (min avg distance)", "algo3_pmedian.png", color=(180, 80, 210))


# P-CENTER 

print("\n── Algorithm 4: P-Center (minimax) ──")
t = time.time()

def pcenter_feasible(r_threshold, p_budget):
    m  = pulp.LpProblem("PCenter", pulp.LpMinimize)
    xc = [pulp.LpVariable(f"xc_{j}", cat='Binary') for j in range(n_cands)]
    m += pulp.lpSum(xc)
    m += pulp.lpSum(xc) <= p_budget
    for i in range(n_demand):
        reachable = [j for j in range(n_cands) if dist_matrix[i][j] <= r_threshold]
        if reachable:
            m += pulp.lpSum(xc[j] for j in reachable) >= 1
    m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
    if pulp.LpStatus[m.status] == 'Optimal':
        return [j for j in range(n_cands) if xc[j].value() and xc[j].value() > 0.5]
    return None

all_dists    = sorted(set(dist_matrix.flatten()))
step_size    = max(1, len(all_dists) // 200)
unique_dists = all_dists[::step_size]
lo, hi       = 0, len(unique_dists) - 1
best_result  = None
iterations   = 0

print(f"    Binary search over {len(unique_dists)} distance thresholds...")
while lo <= hi and iterations < 15:
    mid = (lo + hi) // 2
    r   = unique_dists[mid]
    res = pcenter_feasible(r, P)
    iterations += 1
    status = "Feasible" if res is not None else "Infeasible"
    print(f"    Iteration {iterations}  radius={r:.1f}px  ->  {status}  ({time.time()-t:.0f}s)")
    if res is not None:
        best_result = res
        hi = mid - 1
    else:
        lo = mid + 1

if best_result is None:
    print("    No feasible solution found — using ILP result as fallback")
    best_result = ilp_chosen

print(f"    Done ({time.time()-t:.1f}s)")
draw_result(best_result, "Algorithm 4 - P-Center (minimax worst case)", "algo4_pcenter.png", color=(210, 130, 30))



total_time = time.time() - t_total
print(f"\n[5/5] All done in {total_time:.0f}s\n")
print(f"""
+--------------------------------------------------------------+
|  SUMMARY  (P={P} stations, S={S}px radius, cell={CELL}px)
+--------------------------------------------------------------+
|  Algo 1 - ILP           ->  algo1_ilp.png
|  Algo 2 - K-Means       ->  algo2_kmeans.png
|  Algo 3 - P-Median      ->  algo3_pmedian.png
|  Algo 4 - P-Center      ->  algo4_pcenter.png
+--------------------------------------------------------------+
|  IF SLOW: increase CELL and THRESHOLD at the top of the file
|    CELL=40, THRESHOLD=80   ->  fast, slightly coarser
|    CELL=60, THRESHOLD=180  ->  very fast
+--------------------------------------------------------------+
""")
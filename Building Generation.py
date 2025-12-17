import random
from datetime import datetime

def generate_100_buildings(n=100, weights=None):
    """
    Generates exactly n buildings with types: 1BHK, 2BHK, 3BHK, Commercial.
    weights controls the probability mix (must sum to 1.0).
    Each run is randomized using current time seed (real-time).
    """
    if weights is None:
        # realistic-ish default mix: mostly 1/2BHK, fewer 3BHK and commercial
        weights = {
            "1BHK": 0.30,
            "2BHK": 0.45,
            "3BHK": 0.15,
            "Commercial": 0.10
        }

    # real-time random seed
    seed = int(datetime.now().timestamp() * 1_000_000)
    rng = random.Random(seed)

    types = list(weights.keys())
    probs = [weights[t] for t in types]

    buildings = []
    counts = {t: 0 for t in types}

    for i in range(1, n + 1):
        btype = rng.choices(types, weights=probs, k=1)[0]
        counts[btype] += 1

        buildings.append({
            "Building_ID": f"THENI_BLD_{i:03d}",
            "Building_Type": btype
        })

    return seed, counts, buildings


# ---- Example run ----
seed, counts, buildings = generate_100_buildings()

print("Seed:", seed)
print("Type counts:", counts)
print("First 15 buildings:")
for row in buildings[:15]:
    print(row)

FEATURES = [
    "elevation",         # elevasi DEMNAS ~8m (m)
    "slope",             # kemiringan lereng (derajat)
    "twi",               # Topographic Wetness Index
    "hand",              # Height Above Nearest Drainage (m) — fitur tambahan
    "dist_river",        # jarak ke sungai terdekat (m)
    "drainage_density",  # kerapatan drainase (km/km²)
    "rainfall",          # curah hujan rata-rata tahunan (mm), BMKG Tunggul Wulung
    "landcover",         # kode tutupan lahan ESA WorldCover 2021
]
LABEL = "banjir"  # 1 = titik banjir, 0 = titik non-banjir

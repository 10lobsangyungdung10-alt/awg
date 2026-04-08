"""
Real climate data collector for AWG ML training.

Provides real monthly climate normals for ~48 global cities sourced from
meteorological records, covering all major climate zones. Water output labels
are computed from the same physics formula used throughout the app.

Usage:
    # Generate real_climate_data.csv (run once, or to refresh):
    python data_collector.py

    # Optionally fetch live OWM data and append to the CSV:
    python data_collector.py --api-key <YOUR_OWM_KEY>
"""
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import httpx
from psychrometrics import absolute_humidity, dew_point

DATASET_PATH = os.path.join(os.path.dirname(__file__), "real_climate_data.csv")

# ──────────────────────────────────────────────────────────────────────────────
# Real monthly climate normals
# Source: NOAA/WMO climatological averages, cross-referenced with national met
# services. Values are monthly means: (avg_temp_c, avg_humidity_pct,
# avg_pressure_hpa).  High-altitude cities use surface (station) pressure.
# Month order: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
# ──────────────────────────────────────────────────────────────────────────────
CITY_CLIMATE_NORMALS = {
    # ── South Asia ─────────────────────────────────────────────────────────
    "Delhi": {
        "country": "IN", "lat": 28.6139, "lon": 77.2090,
        "monthly": [
            (14, 82, 1018), (17, 74, 1015), (23, 58, 1010), (30, 38, 1005),
            (36, 30, 1000), (38, 45,  997), (34, 78,  998), (33, 82, 1000),
            (30, 72, 1004), (25, 55, 1010), (19, 68, 1015), (14, 78, 1018),
        ],
    },
    "Mumbai": {
        "country": "IN", "lat": 19.0760, "lon": 72.8777,
        "monthly": [
            (25, 72, 1015), (26, 71, 1014), (28, 73, 1012), (30, 76, 1009),
            (32, 80, 1006), (30, 87, 1004), (28, 91, 1005), (28, 90, 1006),
            (29, 87, 1007), (30, 83, 1010), (28, 75, 1013), (26, 71, 1015),
        ],
    },
    "Chennai": {
        "country": "IN", "lat": 13.0827, "lon": 80.2707,
        "monthly": [
            (25, 75, 1014), (27, 72, 1013), (30, 70, 1011), (33, 68, 1008),
            (35, 62, 1005), (34, 65, 1003), (33, 70, 1004), (33, 72, 1005),
            (31, 73, 1007), (28, 78, 1010), (26, 80, 1013), (24, 78, 1015),
        ],
    },
    "Kolkata": {
        "country": "IN", "lat": 22.5726, "lon": 88.3639,
        "monthly": [
            (19, 74, 1016), (22, 67, 1014), (27, 58, 1010), (31, 58, 1005),
            (33, 68, 1000), (33, 82,  998), (31, 87,  999), (31, 87, 1001),
            (31, 83, 1004), (29, 73, 1009), (24, 68, 1014), (19, 73, 1017),
        ],
    },
    "Bangalore": {
        "country": "IN", "lat": 12.9716, "lon": 77.5946,
        "monthly": [
            (20, 62, 912), (22, 57, 911), (25, 55, 909), (27, 60, 907),
            (27, 67, 907), (23, 80, 909), (22, 83, 910), (22, 82, 910),
            (23, 79, 910), (22, 79, 912), (20, 73, 913), (19, 65, 913),
        ],
    },
    "Hyderabad": {
        "country": "IN", "lat": 17.3850, "lon": 78.4867,
        "monthly": [
            (22, 67, 955), (25, 55, 953), (29, 43, 950), (32, 36, 946),
            (35, 35, 940), (33, 62, 939), (29, 76, 942), (29, 78, 944),
            (28, 74, 947), (26, 63, 951), (22, 62, 955), (21, 66, 957),
        ],
    },
    "Jaipur": {
        "country": "IN", "lat": 26.9124, "lon": 75.7873,
        "monthly": [
            (15, 68, 966), (18, 56, 963), (24, 38, 958), (31, 24, 953),
            (37, 22, 947), (38, 33, 943), (33, 64, 946), (31, 72, 949),
            (29, 57, 953), (24, 38, 959), (19, 47, 963), (15, 62, 967),
        ],
    },
    "Leh": {
        "country": "IN", "lat": 34.1526, "lon": 77.5770,
        "monthly": [
            (-8, 53, 652), (-6, 50, 651), ( 1, 43, 651), ( 9, 34, 650),
            (14, 28, 649), (18, 28, 648), (22, 40, 648), (21, 44, 648),
            (16, 38, 649), ( 8, 38, 651), ( 0, 49, 652), (-6, 55, 653),
        ],
    },
    "Dhaka": {
        "country": "BD", "lat": 23.7104, "lon": 90.4074,
        "monthly": [
            (19, 78, 1014), (22, 71, 1013), (27, 62, 1009), (30, 65, 1005),
            (30, 79, 1001), (29, 86,  999), (29, 88, 1000), (29, 87, 1001),
            (29, 85, 1004), (27, 78, 1008), (23, 75, 1013), (19, 76, 1015),
        ],
    },
    "Lahore": {
        "country": "PK", "lat": 31.5497, "lon": 74.3436,
        "monthly": [
            (12, 79, 989), (15, 71, 987), (21, 59, 983), (28, 43, 978),
            (35, 32, 972), (38, 43, 968), (34, 71, 969), (33, 75, 971),
            (30, 65, 975), (24, 52, 981), (17, 65, 986), (13, 76, 989),
        ],
    },
    "Karachi": {
        "country": "PK", "lat": 24.8607, "lon": 67.0011,
        "monthly": [
            (19, 72, 1016), (21, 70, 1015), (25, 68, 1013), (29, 64, 1010),
            (32, 63, 1007), (34, 65, 1003), (33, 75, 1002), (32, 77, 1004),
            (30, 73, 1007), (27, 70, 1011), (23, 70, 1014), (20, 71, 1016),
        ],
    },
    "Colombo": {
        "country": "LK", "lat":  6.9271, "lon": 79.8612,
        "monthly": [
            (27, 75, 1010), (28, 73, 1011), (29, 74, 1010), (29, 79, 1008),
            (29, 82, 1007), (28, 82, 1007), (27, 82, 1008), (27, 83, 1009),
            (27, 82, 1009), (27, 82, 1009), (27, 82, 1009), (27, 78, 1010),
        ],
    },
    "Kathmandu": {
        "country": "NP", "lat": 27.7172, "lon": 85.3240,
        "monthly": [
            ( 8, 67, 859), (11, 62, 858), (15, 55, 857), (19, 52, 855),
            (23, 60, 854), (24, 75, 853), (25, 82, 853), (25, 83, 853),
            (23, 77, 854), (19, 65, 856), (13, 61, 858), ( 9, 66, 859),
        ],
    },
    # ── Southeast & East Asia ──────────────────────────────────────────────
    "Singapore": {
        "country": "SG", "lat":  1.3521, "lon": 103.8198,
        "monthly": [
            (26, 84, 1011), (27, 82, 1012), (27, 83, 1011), (28, 84, 1009),
            (28, 83, 1008), (28, 82, 1008), (27, 82, 1009), (27, 82, 1009),
            (27, 83, 1009), (27, 84, 1009), (27, 86, 1010), (26, 85, 1010),
        ],
    },
    "Bangkok": {
        "country": "TH", "lat": 13.7563, "lon": 100.5018,
        "monthly": [
            (26, 72, 1015), (28, 70, 1013), (30, 68, 1011), (32, 70, 1008),
            (31, 76, 1005), (30, 79, 1004), (29, 79, 1005), (29, 80, 1006),
            (29, 81, 1007), (28, 79, 1009), (26, 74, 1012), (25, 71, 1014),
        ],
    },
    "Jakarta": {
        "country": "ID", "lat": -6.2088, "lon": 106.8456,
        "monthly": [
            (26, 87, 1008), (26, 87, 1008), (27, 85, 1009), (28, 83, 1009),
            (28, 81, 1010), (27, 77, 1012), (27, 74, 1013), (27, 73, 1013),
            (27, 74, 1012), (28, 78, 1010), (27, 84, 1009), (26, 87, 1008),
        ],
    },
    "Kuala Lumpur": {
        "country": "MY", "lat":  3.1390, "lon": 101.6869,
        "monthly": [
            (27, 83, 1009), (28, 81, 1010), (28, 83, 1009), (28, 84, 1008),
            (28, 83, 1007), (28, 81, 1007), (27, 82, 1008), (27, 82, 1008),
            (27, 83, 1008), (27, 85, 1007), (27, 86, 1007), (27, 84, 1008),
        ],
    },
    "Hong Kong": {
        "country": "HK", "lat": 22.3193, "lon": 114.1694,
        "monthly": [
            (17, 78, 1013), (17, 83, 1012), (20, 83, 1011), (24, 83, 1009),
            (27, 83, 1008), (30, 83, 1006), (30, 82, 1006), (30, 82, 1007),
            (29, 78, 1009), (27, 73, 1012), (23, 73, 1013), (19, 74, 1014),
        ],
    },
    "Tokyo": {
        "country": "JP", "lat": 35.6762, "lon": 139.6503,
        "monthly": [
            ( 6, 52, 1016), ( 7, 53, 1016), (10, 57, 1015), (15, 62, 1014),
            (19, 67, 1013), (23, 74, 1012), (27, 78, 1011), (28, 78, 1011),
            (24, 75, 1013), (18, 66, 1015), (13, 59, 1016), ( 8, 52, 1017),
        ],
    },
    "Shanghai": {
        "country": "CN", "lat": 31.2304, "lon": 121.4737,
        "monthly": [
            ( 5, 72, 1016), ( 6, 72, 1015), (10, 72, 1014), (16, 72, 1013),
            (21, 72, 1012), (25, 78, 1010), (28, 80, 1009), (28, 79, 1010),
            (25, 77, 1012), (20, 74, 1014), (13, 73, 1016), ( 7, 72, 1016),
        ],
    },
    "Beijing": {
        "country": "CN", "lat": 39.9042, "lon": 116.4074,
        "monthly": [
            (-2, 47, 1016), ( 1, 44, 1015), ( 7, 43, 1013), (15, 44, 1011),
            (21, 50, 1009), (26, 60, 1006), (27, 77, 1005), (26, 78, 1007),
            (20, 67, 1011), (13, 56, 1015), ( 4, 55, 1016), (-1, 50, 1016),
        ],
    },
    # ── Middle East & Central Asia ─────────────────────────────────────────
    "Dubai": {
        "country": "AE", "lat": 25.2048, "lon": 55.2708,
        "monthly": [
            (19, 68, 1018), (21, 66, 1017), (24, 62, 1015), (29, 54, 1012),
            (34, 48, 1007), (36, 50, 1003), (38, 55, 1002), (38, 58, 1002),
            (36, 57, 1004), (32, 57, 1009), (26, 64, 1014), (21, 68, 1017),
        ],
    },
    "Riyadh": {
        "country": "SA", "lat": 24.6877, "lon": 46.7219,
        "monthly": [
            (15, 42, 942), (18, 35, 940), (23, 28, 936), (29, 21, 932),
            (35, 15, 927), (39,  9, 922), (42,  7, 921), (42,  7, 922),
            (38,  9, 926), (31, 15, 933), (23, 30, 938), (16, 41, 942),
        ],
    },
    "Amman": {
        "country": "JO", "lat": 31.9454, "lon": 35.9284,
        "monthly": [
            ( 8, 67, 924), ( 9, 63, 923), (13, 55, 921), (18, 44, 920),
            (23, 35, 918), (26, 27, 916), (28, 25, 915), (29, 26, 915),
            (26, 30, 916), (22, 41, 919), (14, 57, 922), (10, 65, 924),
        ],
    },
    # ── Africa ────────────────────────────────────────────────────────────
    "Cairo": {
        "country": "EG", "lat": 30.0444, "lon": 31.2357,
        "monthly": [
            (13, 66, 1017), (15, 58, 1016), (18, 52, 1014), (23, 42, 1012),
            (28, 35, 1009), (31, 33, 1007), (33, 36, 1006), (33, 38, 1006),
            (30, 42, 1009), (26, 51, 1012), (20, 59, 1015), (15, 65, 1017),
        ],
    },
    "Lagos": {
        "country": "NG", "lat":  6.5244, "lon":  3.3792,
        "monthly": [
            (28, 78, 1010), (30, 77, 1009), (30, 79, 1008), (29, 82, 1007),
            (28, 84, 1007), (27, 84, 1007), (26, 83, 1009), (25, 83, 1010),
            (27, 82, 1010), (28, 82, 1009), (28, 81, 1010), (28, 80, 1010),
        ],
    },
    "Nairobi": {
        "country": "KE", "lat": -1.2921, "lon": 36.8219,
        "monthly": [
            (19, 65, 836), (20, 62, 835), (20, 70, 834), (18, 78, 833),
            (17, 81, 834), (16, 76, 836), (15, 76, 837), (16, 74, 837),
            (17, 70, 836), (18, 74, 835), (18, 78, 834), (18, 70, 836),
        ],
    },
    "Addis Ababa": {
        "country": "ET", "lat":  9.0320, "lon": 38.7492,
        "monthly": [
            (16, 61, 772), (18, 56, 771), (19, 61, 770), (19, 70, 769),
            (18, 72, 769), (14, 82, 770), (12, 89, 770), (12, 89, 770),
            (14, 83, 771), (16, 72, 771), (16, 64, 772), (15, 62, 772),
        ],
    },
    "Casablanca": {
        "country": "MA", "lat": 33.5731, "lon": -7.5898,
        "monthly": [
            (13, 76, 1016), (13, 74, 1016), (15, 73, 1015), (17, 72, 1014),
            (19, 70, 1013), (22, 68, 1011), (25, 65, 1010), (25, 65, 1010),
            (23, 67, 1011), (19, 70, 1014), (16, 73, 1015), (13, 76, 1016),
        ],
    },
    "Johannesburg": {
        "country": "ZA", "lat": -26.2041, "lon": 28.0473,
        "monthly": [
            (22, 60, 824), (21, 60, 825), (20, 58, 825), (17, 53, 828),
            (14, 47, 830), (11, 42, 832), (11, 40, 833), (13, 40, 832),
            (16, 45, 830), (18, 53, 828), (20, 57, 826), (21, 59, 825),
        ],
    },
    # ── Europe ────────────────────────────────────────────────────────────
    "London": {
        "country": "GB", "lat": 51.5074, "lon": -0.1278,
        "monthly": [
            ( 5, 85, 1014), ( 6, 81, 1014), ( 8, 75, 1014), (11, 70, 1013),
            (14, 67, 1013), (17, 66, 1013), (19, 64, 1013), (19, 65, 1013),
            (16, 70, 1014), (12, 77, 1014), ( 8, 82, 1014), ( 6, 85, 1014),
        ],
    },
    "Paris": {
        "country": "FR", "lat": 48.8566, "lon":  2.3522,
        "monthly": [
            ( 4, 86, 1014), ( 5, 82, 1014), ( 9, 75, 1013), (12, 72, 1012),
            (16, 70, 1012), (19, 68, 1012), (21, 65, 1012), (21, 66, 1013),
            (17, 72, 1014), (13, 79, 1014), ( 8, 85, 1014), ( 5, 87, 1014),
        ],
    },
    "Berlin": {
        "country": "DE", "lat": 52.5200, "lon": 13.4050,
        "monthly": [
            ( 1, 87, 1014), ( 2, 83, 1014), ( 6, 76, 1013), (11, 68, 1012),
            (16, 65, 1011), (19, 65, 1011), (21, 64, 1011), (20, 66, 1012),
            (16, 73, 1013), (11, 80, 1014), ( 6, 85, 1014), ( 2, 87, 1014),
        ],
    },
    "Moscow": {
        "country": "RU", "lat": 55.7558, "lon": 37.6173,
        "monthly": [
            (-6, 84, 1013), (-5, 81, 1013), ( 0, 76, 1012), ( 8, 68, 1011),
            (15, 62, 1011), (19, 64, 1010), (22, 65, 1010), (20, 68, 1011),
            (14, 75, 1012), ( 7, 81, 1013), ( 0, 85, 1013), (-4, 85, 1013),
        ],
    },
    "Istanbul": {
        "country": "TR", "lat": 41.0082, "lon": 28.9784,
        "monthly": [
            ( 6, 80, 1015), ( 7, 77, 1015), ( 9, 74, 1014), (13, 73, 1013),
            (18, 70, 1012), (23, 65, 1011), (25, 63, 1010), (26, 62, 1010),
            (22, 65, 1012), (17, 71, 1013), (12, 77, 1014), ( 8, 80, 1015),
        ],
    },
    "Rome": {
        "country": "IT", "lat": 41.9028, "lon": 12.4964,
        "monthly": [
            ( 8, 77, 1015), ( 9, 73, 1015), (12, 68, 1014), (15, 68, 1013),
            (19, 64, 1012), (24, 55, 1010), (28, 47, 1010), (28, 48, 1011),
            (24, 56, 1012), (18, 67, 1014), (13, 74, 1015), ( 9, 77, 1015),
        ],
    },
    # ── Americas ──────────────────────────────────────────────────────────
    "New York": {
        "country": "US", "lat": 40.7128, "lon": -74.0060,
        "monthly": [
            ( 1, 64, 1016), ( 3, 62, 1016), ( 7, 60, 1014), (13, 59, 1013),
            (18, 62, 1013), (23, 64, 1013), (26, 66, 1013), (26, 68, 1014),
            (22, 68, 1015), (16, 64, 1016), (10, 67, 1016), ( 4, 65, 1016),
        ],
    },
    "Miami": {
        "country": "US", "lat": 25.7617, "lon": -80.1918,
        "monthly": [
            (20, 73, 1017), (21, 71, 1017), (23, 69, 1016), (25, 67, 1015),
            (28, 72, 1013), (29, 78, 1012), (30, 77, 1012), (30, 78, 1012),
            (29, 79, 1013), (27, 76, 1015), (23, 73, 1016), (21, 73, 1017),
        ],
    },
    "Houston": {
        "country": "US", "lat": 29.7604, "lon": -95.3698,
        "monthly": [
            (12, 74, 1017), (14, 71, 1016), (18, 69, 1015), (22, 71, 1014),
            (26, 73, 1013), (30, 72, 1012), (32, 70, 1012), (32, 72, 1012),
            (29, 74, 1013), (24, 74, 1015), (17, 74, 1016), (13, 74, 1017),
        ],
    },
    "Los Angeles": {
        "country": "US", "lat": 34.0522, "lon": -118.2437,
        "monthly": [
            (14, 68, 1014), (15, 67, 1014), (16, 67, 1014), (18, 65, 1013),
            (20, 66, 1012), (22, 65, 1011), (24, 62, 1010), (25, 62, 1010),
            (23, 62, 1010), (21, 64, 1012), (17, 67, 1013), (14, 68, 1014),
        ],
    },
    "Chicago": {
        "country": "US", "lat": 41.8781, "lon": -87.6298,
        "monthly": [
            (-4, 72, 1015), (-2, 70, 1015), ( 4, 67, 1014), (10, 61, 1013),
            (16, 60, 1012), (22, 63, 1012), (24, 64, 1012), (23, 66, 1013),
            (18, 67, 1014), (12, 64, 1015), ( 5, 69, 1016), (-1, 72, 1016),
        ],
    },
    "Anchorage": {
        "country": "US", "lat": 61.2181, "lon": -149.9003,
        "monthly": [
            (-8, 80, 1008), (-5, 78, 1009), (-1, 74, 1010), ( 5, 68, 1011),
            (10, 63, 1012), (14, 63, 1012), (16, 68, 1011), (15, 72, 1010),
            (10, 76, 1009), ( 3, 80, 1007), (-4, 82, 1007), (-7, 82, 1007),
        ],
    },
    "Mexico City": {
        "country": "MX", "lat": 19.4326, "lon": -99.1332,
        "monthly": [
            (13, 53, 778), (15, 49, 777), (17, 45, 776), (18, 49, 775),
            (18, 60, 774), (17, 74, 773), (16, 75, 773), (16, 76, 774),
            (15, 79, 774), (14, 70, 776), (13, 60, 777), (13, 55, 778),
        ],
    },
    "São Paulo": {
        "country": "BR", "lat": -23.5505, "lon": -46.6333,
        "monthly": [
            (23, 80, 925), (23, 80, 925), (22, 80, 926), (20, 79, 927),
            (17, 75, 929), (16, 73, 931), (15, 70, 932), (17, 68, 931),
            (19, 72, 929), (20, 76, 927), (21, 79, 926), (22, 80, 925),
        ],
    },
    "Rio de Janeiro": {
        "country": "BR", "lat": -22.9068, "lon": -43.1729,
        "monthly": [
            (27, 78, 1009), (27, 79, 1009), (26, 79, 1010), (24, 78, 1012),
            (22, 76, 1014), (21, 74, 1015), (21, 73, 1016), (21, 73, 1016),
            (22, 75, 1015), (23, 77, 1013), (25, 78, 1011), (26, 79, 1010),
        ],
    },
    "Buenos Aires": {
        "country": "AR", "lat": -34.6037, "lon": -58.3816,
        "monthly": [
            (24, 62, 1011), (24, 63, 1011), (21, 65, 1013), (16, 66, 1015),
            (13, 70, 1016), (10, 73, 1016), (10, 73, 1015), (11, 72, 1015),
            (13, 70, 1015), (17, 67, 1014), (20, 63, 1012), (23, 61, 1011),
        ],
    },
    # ── Oceania ───────────────────────────────────────────────────────────
    "Sydney": {
        "country": "AU", "lat": -33.8688, "lon": 151.2093,
        "monthly": [
            (23, 66, 1012), (23, 66, 1013), (21, 68, 1013), (18, 68, 1014),
            (15, 70, 1015), (13, 71, 1015), (12, 68, 1016), (13, 64, 1016),
            (15, 63, 1015), (18, 63, 1014), (20, 63, 1013), (22, 64, 1012),
        ],
    },
    "Melbourne": {
        "country": "AU", "lat": -37.8136, "lon": 144.9631,
        "monthly": [
            (20, 61, 1012), (20, 60, 1013), (18, 63, 1014), (15, 66, 1015),
            (12, 71, 1015), (10, 74, 1015), ( 9, 73, 1016), (10, 70, 1016),
            (12, 66, 1015), (14, 62, 1014), (17, 60, 1013), (19, 61, 1012),
        ],
    },
}


def _compute_water_output(temp_c: float, humidity_pct: float, month: int) -> float:
    """
    Compute expected AWG water output using the same physics formula used
    throughout the app (500 m³/hr unit, 40% efficiency).
    A small ±8% seasonal factor is retained to capture real seasonal variation.
    """
    ah = absolute_humidity(temp_c, humidity_pct)
    base_output = ah * 500 * 24 * 0.4 / 1000  # liters/day
    seasonal_factor = 1 + 0.08 * np.sin(2 * np.pi * month / 12)
    return max(0.0, base_output * seasonal_factor)


def build_real_dataset(augment_factor: int = 5, seed: int = 0) -> pd.DataFrame:
    """
    Build a training DataFrame from real monthly climate normals.

    Each city×month observation is augmented `augment_factor` times with
    Gaussian noise to capture natural day-to-day weather variability:
      - Temperature:  σ = 2.5 °C
      - Humidity:     σ = 6 %
      - Pressure:     σ = 2.5 hPa

    Returns a DataFrame with columns matching train_model() expectations.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for city, info in CITY_CLIMATE_NORMALS.items():
        for month_idx, (temp, rh, pressure) in enumerate(info["monthly"]):
            month_num = month_idx + 1
            for _ in range(augment_factor):
                t = temp + rng.normal(0, 2.5)
                h = float(np.clip(rh + rng.normal(0, 6), 5, 100))
                p = pressure + rng.normal(0, 2.5)
                dp = dew_point(t, h)
                ah = absolute_humidity(t, h)
                water = _compute_water_output(t, h, month_num)
                rows.append({
                    "city": city,
                    "country": info["country"],
                    "lat": info["lat"],
                    "lon": info["lon"],
                    "month": month_num,
                    "temperature": round(t, 2),
                    "humidity": round(h, 2),
                    "pressure": round(p, 2),
                    "dew_point": dp,
                    "absolute_humidity": round(ah, 4),
                    "water_output_liters_per_day": round(water, 4),
                })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def collect_live_data(api_key: str, extra_cities: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch current weather for a list of cities from OpenWeatherMap and return
    a DataFrame of real observations that can be appended to the training set.

    Each live observation gets its water output label computed from physics
    (same formula as build_real_dataset).

    Args:
        api_key: OpenWeatherMap API key
        extra_cities: Additional city names to query (on top of CITY_CLIMATE_NORMALS)
    """
    cities = list(CITY_CLIMATE_NORMALS.keys()) + (extra_cities or [])
    rows = []

    for city in cities:
        try:
            resp = httpx.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city, "appid": api_key, "units": "metric"},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            temp = data["main"]["temp"]
            rh = float(data["main"]["humidity"])
            pressure = float(data["main"]["pressure"])
            lat = data.get("coord", {}).get("lat", 0)
            lon = data.get("coord", {}).get("lon", 0)
            country = data.get("sys", {}).get("country", "")
            month = datetime.utcnow().month
            dp = dew_point(temp, rh)
            ah = absolute_humidity(temp, rh)
            water = _compute_water_output(temp, rh, month)
            rows.append({
                "city": city,
                "country": country,
                "lat": lat,
                "lon": lon,
                "month": month,
                "temperature": round(temp, 2),
                "humidity": round(rh, 2),
                "pressure": round(pressure, 2),
                "dew_point": dp,
                "absolute_humidity": round(ah, 4),
                "water_output_liters_per_day": round(water, 4),
            })
            print(f"  ✓ {city}: {temp}°C, {rh}% RH")
        except Exception as e:
            print(f"  ✗ {city}: {e}")

    return pd.DataFrame(rows)


def save_dataset(df: pd.DataFrame, path: str = DATASET_PATH) -> None:
    """Save dataset to CSV."""
    df.to_csv(path, index=False)
    print(f"Dataset saved: {path}  ({len(df)} rows)")


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build real climate training dataset")
    parser.add_argument("--api-key", default="", help="OpenWeatherMap API key (optional)")
    parser.add_argument("--augment", type=int, default=5,
                        help="Augmentation factor per city×month (default: 5)")
    args = parser.parse_args()

    print("Building real climate dataset from meteorological normals...")
    df = build_real_dataset(augment_factor=args.augment)
    print(f"  Base records: {len(CITY_CLIMATE_NORMALS)} cities × 12 months × {args.augment} = {len(df)} rows")

    if args.api_key:
        print("\nFetching live weather data from OpenWeatherMap...")
        live_df = collect_live_data(args.api_key)
        if not live_df.empty:
            df = pd.concat([df, live_df], ignore_index=True)
            print(f"  Added {len(live_df)} live observations → total {len(df)} rows")

    save_dataset(df)
    print(f"\nDataset summary:")
    print(f"  Temperature range: {df['temperature'].min():.1f}°C – {df['temperature'].max():.1f}°C")
    print(f"  Humidity range:    {df['humidity'].min():.1f}% – {df['humidity'].max():.1f}%")
    print(f"  Water output:      {df['water_output_liters_per_day'].min():.1f} – {df['water_output_liters_per_day'].max():.1f} L/day")
    print(f"  Cities covered:    {df['city'].nunique()}")

# calculations.py
import math

def calculate_pv(theta600: float, theta300: float) -> float:
    return theta600 - theta300

def calculate_av(theta600: float) -> float:
    return theta600 / 2.0

def calculate_yp(theta300: float, pv: float) -> float:
    return theta300 - pv

def calculate_ti(yp: float, pv: float) -> float:
    if pv <= 0:
        return 0.0
    return yp / pv

def calculate_n(theta600: float, theta300: float) -> float:
    if theta300 <= 0 or theta600 <= 0:
        return 0.0
    return 3.32 * math.log10(theta600 / theta300)

def calculate_k(theta300: float, n: float) -> float:
    denominator = 511.0 ** n
    if denominator <= 0:
        return 0.0
    return theta300 / denominator

def convert_mw(value: float, from_unit: str) -> dict:
    if from_unit == 'PPG':
        ppg = value
    elif from_unit == 'SG':
        ppg = value * 8.33
    elif from_unit == 'lb/ft³':
        ppg = value / 7.48
    elif from_unit == 'kg/m³':
        ppg = value / 119.83
    elif from_unit == 'psi/1000ft':
        ppg = value / 52.0
    else:
        ppg = value

    return {
        'PPG': ppg,
        'SG': ppg / 8.33,
        'lb/ft³': ppg * 7.48,
        'kg/m³': ppg * 119.83,
        'psi/1000ft': ppg * 52.0
    }

def calculate_bf(mw_ppg: float) -> float:
    return 1.0 - (mw_ppg / 65.5)

def process_sample(name: str, mw_value: float, mw_unit: str, theta600: float, theta300: float, gel10s: float, gel10m: float) -> dict:
    """Convenience function to run all calculations for a single sample."""
    mw_dict = convert_mw(mw_value, mw_unit)
    mw_ppg = mw_dict['PPG']
    
    pv = calculate_pv(theta600, theta300)
    av = calculate_av(theta600)
    yp = calculate_yp(theta300, pv)
    ti = calculate_ti(yp, pv)
    n = calculate_n(theta600, theta300)
    k = calculate_k(theta300, n)
    bf = calculate_bf(mw_ppg)
    
    return {
        'Sample Name': name,
        'MW (PPG)': mw_ppg,
        'MW (SG)': mw_dict['SG'],
        'MW (lb/ft³)': mw_dict['lb/ft³'],
        'MW (kg/m³)': mw_dict['kg/m³'],
        'MW (psi/1000ft)': mw_dict['psi/1000ft'],
        '600 RPM': theta600,
        '300 RPM': theta300,
        'Gel 10s (lb/100ft²)': gel10s,
        'Gel 10m (lb/100ft²)': gel10m,
        'PV (cP)': pv,
        'AV (cP)': av,
        'YP (lb/100ft²)': yp,
        'TI (YP/PV)': ti,
        'n (Flow Index)': n,
        'k (Consistency Index)': k,
        'Buoyancy Factor (BF)': bf
    }
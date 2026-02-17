import numpy as np
import pandas as pd

def generate_engine_data(rows=100000, random_state=42):
    np.random.seed(random_state)

    rpm = np.random.normal(2000, 300, rows)
    temperature = np.random.normal(90, 10, rows)
    fuel_rate = np.random.normal(15, 2, rows)
    oil_pressure = np.random.normal(40, 5, rows)

    anomaly_indices = np.random.choice(rows, size=int(0.05 * rows), replace=False)

    rpm[anomaly_indices] *= np.random.uniform(1.5, 2.0, len(anomaly_indices))
    temperature[anomaly_indices] *= np.random.uniform(1.3, 1.8, len(anomaly_indices))
    fuel_rate[anomaly_indices] *= np.random.uniform(1.4, 1.9, len(anomaly_indices))

    df = pd.DataFrame({
        "rpm": rpm,
        "temperature": temperature,
        "fuel_rate": fuel_rate,
        "oil_pressure": oil_pressure
    })

    return df

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

days_hist = np.array([0, 1, 2, 3, 4, 5, 9, 10, 11]) 

moisture_hist = np.array([79.83, 78.15, 73.56, 59.90, 61.43, 66.65, 75.80, 74.43, 68.45])

CRITICAL_THRESHOLD = 35.0
WARNING_THRESHOLD = 48.0

print("--- MEMULAI PROSES LEARNING ---")

def model_decay_physics(t, M0, k):
    decay = M0 * np.exp(-k * t)
    diurnal = 0.5 * np.sin(2 * np.pi * t) 
    return decay + diurnal


drying_days = []
drying_moistures = []

for i in range(len(days_hist)-1):
    if moisture_hist[i+1] < moisture_hist[i]:
        drying_days.append(days_hist[i+1] - days_hist[i]) # dt
        ratio = moisture_hist[i+1] / moisture_hist[i]
        # k = -ln(ratio)/dt
        k_inst = -np.log(ratio) / (days_hist[i+1] - days_hist[i])
        drying_moistures.append(k_inst)

LEARNED_K = np.mean(drying_moistures)

print(f"Data Historis Dianalisis: {len(days_hist)} titik.")
print(f"Laju Pengeringan Efektif (Learned k): {LEARNED_K:.4f} / hari")
print("-" * 40)

last_day = days_hist[-1]
last_moisture = moisture_hist[-1]

days_future = np.linspace(last_day, last_day + 7, 200)

# M(t) = M_last * exp(-k * delta_t)
delta_t = days_future - last_day
moisture_future = last_moisture * np.exp(-LEARNED_K * delta_t)

# efek osilasi harian pada prediksi biar konsisten
moisture_future += 0.5 * np.sin(2 * np.pi * days_future)

critical_indices = np.where(moisture_future <= CRITICAL_THRESHOLD)[0]

t_critical = None
if len(critical_indices) > 0:
    idx = critical_indices[0]
    t_critical = days_future[idx]
    print(f"[PREDIKSI] Titik Kritis ({CRITICAL_THRESHOLD}%) akan tercapai pada Hari ke-{t_critical:.2f}")
    print(f"           ({t_critical - last_day:.1f} hari dari data terakhir)")
else:
    print("[PREDIKSI] Aman. Tidak mencapai titik kritis dalam 7 hari ke depan.")

plt.figure(figsize=(12, 7))

plt.plot(days_hist, moisture_hist, 'ko', markersize=8, label='Data Historis (Observasi)', zorder=5)
plt.plot(days_hist, moisture_hist, 'k:', alpha=0.3)

t_hist_smooth = np.linspace(0, last_day, 100)
m_hist_model = moisture_hist[0] * np.exp(-LEARNED_K * t_hist_smooth)
plt.plot(t_hist_smooth, m_hist_model, 'b-', alpha=0.4, label=f'Model Regresi (Learned k={LEARNED_K:.3f})')

plt.plot(days_future, moisture_future, 'r--', linewidth=2.5, label='Prediksi 7 Hari (Forecasting)')

plt.axhline(y=CRITICAL_THRESHOLD, color='darkred', linestyle='-', linewidth=2, label=f'Batas Kritis Api ({CRITICAL_THRESHOLD}%)')
plt.axhline(y=WARNING_THRESHOLD, color='orange', linestyle='--', label='Batas Waspada')

plt.fill_between(days_future, 0, CRITICAL_THRESHOLD, color='red', alpha=0.1) # Zona Bahaya

if t_critical:
    plt.plot(t_critical, CRITICAL_THRESHOLD, 'X', color='red', markersize=15, markeredgecolor='black', zorder=10)
    plt.annotate(f'BAHAYA!\nHari ke-{t_critical:.1f}', 
                 xy=(t_critical, CRITICAL_THRESHOLD), 
                 xytext=(t_critical, CRITICAL_THRESHOLD+10),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=11, fontweight='bold', color='red', ha='center')

plt.title(f'Analisis Data Historis & Prediksi Titik Kritis Kebakaran Lahan Gambut\n(Start Prediksi: Hari ke-{last_day} | Moisture Awal: {last_moisture:.1f}%)', fontsize=14)
plt.xlabel('Waktu (Hari)', fontsize=12)
plt.ylabel('Kadar Air Tanah (%)', fontsize=12)
plt.ylim(0, 100)
plt.xlim(0, last_day + 7.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', frameon=True, shadow=True)

plt.axvline(x=last_day, color='gray', linestyle='--')
plt.text(last_day, 95, ' SEKARANG (Last Data) ', ha='center', va='center', backgroundcolor='white', bbox=dict(facecolor='white', edgecolor='gray'))

plt.tight_layout()
plt.show()

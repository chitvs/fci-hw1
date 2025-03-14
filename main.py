import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.fft import fft, fftfreq, ifft, fftshift

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# 1) Definizione del segnale e grafici

f1 = scipy.io.loadmat("eeg_CP4_MI_LH_s01.mat")
signal1 = f1['eeg_CP4_MI_LH_s01'][0]

n1 = 500
n2 = 2500
t = 0.002
T = (n2 - n1)*t

x = np.linspace(n1*t, n2*t, n2-n1)

x_n = signal1[n1:n2]

plt.figure(figsize=(10, 5))
plt.plot(x, x_n , linewidth=0.9)
plt.xlabel("Tempo (s)", fontsize=14)
plt.ylabel("Ampiezza", fontsize=14)

def energia(signal):
    return np.sum(signal ** 2)

energia_y = energia(x_n)

plt.text(0.05, 0.9, r'$\mathrm{Energia} = %.2f$' % energia_y,
         transform=plt.gca().transAxes, fontsize=16, color='black',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3', linewidth=1))
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) Studio di una coppia di segnali

f2 = scipy.io.loadmat("eeg_C4_MI_LH_s01.mat")
signal2 = f2['eeg_C4_MI_LH_s01'][0]

y_n_originale = signal2[n1:n2]

def valore_medio(signal):
    return np.mean(signal)

val_medio = valore_medio(y_n_originale)
y_n = y_n_originale - val_medio

plt.figure(figsize=(10, 5))
plt.plot(x, y_n, linewidth=0.9)
plt.xlabel("Tempo (s)", fontsize=14)
plt.ylabel("Ampiezza", fontsize=14)
plt.text(0.05, 0.9, r'$\mathrm{Valore \; medio} = %f$' % val_medio,
         transform=plt.gca().transAxes, fontsize=16, color='black',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3', linewidth=1))
plt.grid(True)
plt.tight_layout()
plt.show()

matrice_corr = np.corrcoef(x_n, y_n)
coeff_corr = matrice_corr[0][1]

# print(f"coefficiente di correlazione: {coeff_corr}")

# 3) Studio del segnale in frequenza

X_f = fft(x_n)
freqs = fftshift(fftfreq(len(X_f), t))
X_f_shifted = fftshift(X_f)

plt.figure(figsize=(10, 5))
plt.plot(freqs, np.abs(X_f_shifted))
plt.xlabel("Frequenza (Hz)", fontsize=14)
plt.ylabel("Ampiezza", fontsize=14)
plt.ticklabel_format(style='plain', axis='y')
plt.grid(True)
plt.tight_layout()
plt.show()

def ideal_filter(sig, freqsig, f_low=None, f_high=None):
    H = np.ones_like(sig)
    if f_low is not None:
        H[np.abs(freqsig) < f_low] = 0
    if f_high is not None:
        H[np.abs(freqsig) > f_high] = 0
    return H

f_c_low = 8
f_c_high = 13

H_BP_f = ideal_filter(X_f, freqs, f_low=f_c_low, f_high=f_c_high)

Y_BP_f = X_f*H_BP_f
y_BP = ifft(Y_BP_f)

plt.figure(figsize=(10, 5))
plt.plot(freqs, H_BP_f)
plt.xlabel("Frequenza (Hz)", fontsize=14)
plt.ylabel("Ampiezza", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(x, y_BP, linewidth=0.7)
plt.xlabel("Tempo (s)", fontsize=14)
plt.ylabel("Ampiezza", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Domanda Extra

f3 = scipy.io.loadmat("eeg_CP4_rest_s01.mat")
signal3 = f3['eeg_CP4_rest_s01'][0]

def energia_media_f(signal, Nf=500):
    val_en_med = []
    num_finestra = len(signal) // Nf  # Calcola il numero totale di finestre intere
    for i in range(num_finestra):
        finestra = signal[i * Nf:(i + 1) * Nf]
        energia_media = energia(finestra) / Nf
        val_en_med.append(energia_media)
    return val_en_med

extra1 = energia_media_f(signal1)
extra3 = energia_media_f(signal3)

asse_x1 = np.arange(len(extra1))
asse_x3 = np.arange(len(extra3))

plt.figure(figsize=(10, 5))
plt.plot(asse_x3, extra3, color = 'blue', label = "Resting state")
plt.plot(asse_x1, extra1, color = 'red', label = "Motor imagery")
plt.xlabel("Numero di finestre", fontsize=14)
plt.ylabel("Energia media", fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show(block=True) 

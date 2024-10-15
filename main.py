import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import zplane as zp
from typing import Literal


def get_aberration(img):
    poles = [0, -0.99, -0.99, 0.8]
    zeroes = [0.9 * np.exp(1j * np.pi / 2), 0.9 * np.exp(-1j * np.pi / 2), 0.95 * np.exp(1j * np.pi / 8),
              0.95 * np.exp(-1j * np.pi / 8)]

    a = np.poly(poles)
    b = np.poly(zeroes)

    zp.zplane(b, a, plot_title="Stabilité du filtre non-corrigé")

    show_image(img, "Image avec les aberrations")

    return b, a


def correct_aberrations(img, b, a):
    corrected_img = signal.lfilter(a, b, img)
    zp.zplane(a, b, plot_title="Stabilité du filtre corrigé")
    show_image(corrected_img, "Image sans les aberrations")
    return corrected_img

def show_image(img, title):
    plt.gray()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

def create_filter(filter_type: Literal["butter", "cheby1", "cheby2", "ellip"]= "butter"):
    fs = 1_600
    fc = 500
    wp = fc / (fs / 2)
    ws = 750 / (fs / 2)
    rp = 0.2
    rs = 60
    if filter_type == "butter":
        N_min, wn = signal.buttord(wp, ws, rp, rs, fs=fs)
        b, a = signal.butter(N_min, wn, btype="low", output='ba')
        print(f"ordre du filtre {filter_type}: {N_min}")
    elif filter_type == "cheby1":
        N_min, wn = signal.cheb1ord(wp, ws, rp, rs, fs=fs)
        b, a = signal.cheby1(N_min, rp, wn, btype="low", output='ba')
        print(f"ordre du filtre {filter_type}: {N_min}")

    elif filter_type == "cheby2":
        N_min, wn = signal.cheb2ord(wp, ws, rp, rs, fs=fs)
        b, a = signal.cheby2(N_min, rp, wn, btype="low", output='ba')
        print(f"ordre du filtre {filter_type}: {N_min}")

    elif filter_type == "ellip":
        N_min, wn = signal.ellipord(wp, ws, rp, rs, fs=fs)
        b, a = signal.ellip(N_min, rp, rs, wn, btype="low", output='ba')
        print(f"ordre du filtre {filter_type}: {N_min}")


    zeroes = np.roots(b)
    poles = np.roots(a)

    w, H = signal.freqz(b, a, fs)
    H_db = 20 * np.log10(np.abs(H))
    phase = np.angle(H)
    # Transforme freq normalise sur pi en Hz
    freq_Hz = w * fs/(2*np.pi)



    match filter_type:
        case "butter":
            filter_type_name = "Butterworth"
        case "cheby1":
            filter_type_name = "Chebyshev Type I"
        case "cheby2":
            filter_type_name = "Chebyshev Type II"
        case "ellip":
            filter_type_name = "Elliptic"
        case _: filter_type_name = "Butterworth"

    zp.zplane(b, a, plot_title=f"Plan Complexe du filtre de type {filter_type_name}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(freq_Hz, H_db)
    ax1.set_title(f"Réponse fréquentielle du filtre passe-bas de type {filter_type_name}")
    ax1.set_xlabel("Fréquence (Hz)")
    ax1.set_ylabel("Amplitude (dB)")
    ax1.grid()

    ax2.plot(freq_Hz, np.unwrap(phase))
    ax2.set_xlabel("Fréquence (Hz)")
    ax2.set_ylabel("Phase (radians)")
    ax2.grid()

    plt.tight_layout()
    plt.show()

    # print(f"ordre min: {N_min}")
    # print(f"zeroes: {zeroes}")
    # print(f"poles: {poles}")

    return b, a, N_min

def apply_filter(img):
    show_image(img, "Image avant le filtrage du bruit")

    filter_types = ["butter", "cheby1", "cheby2", "ellip"]

    for filter_type in filter_types:
        match filter_type:
            case "butter":
                filter_type_name = "Butterworth"
            case "cheby1":
                filter_type_name = "Chebyshev Type I"
            case "cheby2":
                filter_type_name = "Chebyshev Type II"
            case "ellip":
                filter_type_name = "Elliptic"
            case _:
                filter_type_name = "Butterworth"
        b, a, N = create_filter(filter_type)
        filtered_img = signal.lfilter(b, a, img)
        show_image(filtered_img, f"Image apres filtrage avec un filtre {filter_type_name} d'ordre {N}")
        if (filter_type == "ellip"):
            return filtered_img


def make_homemade_butterworth(img):

    fe = 1600
    wc = 500
    freq_gauchichessement = 2*fe*np.tan((np.pi*wc)/fe)
    print(f"Fréquence gauchissement: {freq_gauchichessement} rad/s")
    freq_hertz = freq_gauchichessement / (2*np.pi)
    print(f"Fréquence gauchissement: {freq_hertz} Hz")

    A = 2 * fe / freq_hertz
    B = np.sqrt(2)
    C = A**2 + A * B + 1

    print(f"A: {A}")
    print(f"C: {C}")

    b = [1/C, 2/C, 1/C]
    a = [1, (-2 * A**2 + 2)/C, (A**2 - A * B + 1)/C]
    print(f"b: {b}")
    print(f"a: {a}")

    w, H = signal.freqz(b, a, fe)
    H_db = 20 * np.log10(np.abs(H))
    phase = np.angle(H)
    freq_Hz = w * fe / (2 * np.pi)

    zp.zplane(b, a, plot_title="Plan Complexe du filtre Butterworth d'ordre 2")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(freq_Hz, H_db)
    ax1.set_title(f"Reponse fréquentielle du filtre passe-bas de type Butterworth d'ordre 2")
    ax1.set_xlabel("Fréquence (Hz)")
    ax1.set_ylabel("Amplitude (dB)")
    ax1.grid()

    ax2.plot(freq_Hz, np.unwrap(phase))
    ax2.set_xlabel("Fréquence (Hz)")
    ax2.set_ylabel("Phase (radians)")
    ax2.grid()

    plt.tight_layout()
    plt.show()

    filtered_img = signal.lfilter(b, a, img)
    show_image(filtered_img, "Image avec le filtre Butterworth d'ordre 2\nconçu avec la transformation bilinéaire")

def rotate_base(img):
    # img = np.mean(img, -1)
    show_image(img, "Image avant la rotation")
    x, y  = img.shape

    new_img = np.zeros((x, y))
    for e1 in range(x):
        for e2 in range(y):
            u1 = e2
            u2 = -e1

            new_img[u1][u2] = img[e1][e2]

    show_image(new_img, "Image après la rotation")

    return new_img

def compress(img, factor=0.5):

    matrice_cov = np.cov(img)
    eigen_values, eigen_vector = np.linalg.eig(matrice_cov)
    transfer_matrix = np.transpose(eigen_vector)
    inversed_transfer_matrix = np.linalg.inv(transfer_matrix)

    Iv = transfer_matrix.dot(img)
    size = len(Iv)
    Iv = [Iv[n] if n < (size * factor) else np.zeros(size) for n in range(size)]
    Io = inversed_transfer_matrix.dot(Iv)

    show_image(Io, f"Image après suppression de {(1-factor)*100}%\ndes vecteurs propres")

    return Io

def extract_complete_img(img_path="./assets/image_complete.npy"):
    img = np.load(img_path)
    b, a = get_aberration(img)
    corrected_aberration = correct_aberrations(img, b, a)
    rotated_img = rotate_base(corrected_aberration)
    elliptic_filter = apply_filter(rotated_img)
    make_homemade_butterworth(rotated_img)
    show_image(elliptic_filter, "Image avant la compression")
    compressed_img = compress(elliptic_filter)
    more_compressed_img = compress(elliptic_filter, 0.3)




if __name__ == '__main__':
    extract_complete_img()
